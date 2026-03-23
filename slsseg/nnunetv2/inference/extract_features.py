import inspect
import itertools
import json
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.hook_manager import FeatureHookManager


class nnUNetFeatureExtractor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm
        )

    def _resolve_case_id(
        self,
        preprocessed: dict,
        fallback_index: int,
    ) -> str:
        """Resolve a stable case identifier for feature saving."""
        ofile = preprocessed.get("ofile")
        if ofile is not None:
            return os.path.basename(ofile)

        properties = preprocessed.get("data_properties", {})
        if properties.get("case_identifier") is not None:
            return str(properties["case_identifier"])

        data_files = properties.get("list_of_data_files")
        if data_files:
            first_file = os.path.basename(data_files[0])
            suffix = self.dataset_json["file_ending"]
            if first_file.endswith(suffix):
                first_file = first_file[:-len(suffix)]
            if first_file.endswith("_0000"):
                first_file = first_file[:-5]
            return first_file

        return f"case_{fallback_index:04d}"

    def predict_logits_and_features_from_preprocessed_data(
        self,
        data: torch.Tensor,
        layer_names: List[str],
        capture: str = "output",
        store_all_calls: bool = True,
    ):
        """Predict logits and collect intermediate features.

        Parameters
        ----------
        data:
            Preprocessed tensor of shape (C, X, Y, Z).
        layer_names:
            Exact module names from `self.network.named_modules()`.
        capture:
            One of {"output", "input"}.
        store_all_calls:
            If True, keep one tensor per sliding-window patch. If False, keep only
            the most recent call for each layer.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]
            Predicted logits and collected feature tensors.
        """
        hook_manager = FeatureHookManager(
            model=self.network,
            layer_names=layer_names,
            capture=capture,
            detach=True,
            move_to_cpu=True,
            store_all_calls=store_all_calls,
        )
        hook_manager.register()

        try:
            logits = self.predict_logits_from_preprocessed_data(data)
            features = hook_manager.get_features()
        finally:
            hook_manager.remove()

        return logits, features

    def extract_features_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder: str,
        layer_names: List[str],
        capture: str = "output",
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        store_all_calls: bool = True,
        save_patch_features: bool = False,
    ):
        """Extract and save hooked features for each case."""
        maybe_mkdir_p(output_folder)

        metadata = {
            "layer_names": layer_names,
            "capture": capture,
            "store_all_calls": store_all_calls,
            "save_patch_features": save_patch_features,
        }
        save_json(metadata, join(output_folder, "feature_extraction_args.json"))
        save_json(self.dataset_json, join(output_folder, "dataset.json"),
                  sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, "plans.json"),
                  sort_keys=False)

        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It ' \
                f'requires the segmentations of the previous stage ' \
                f'({self.configuration_manager.previous_stage_name}) as ' \
                f'input. Please provide the folder where they are located ' \
                f'via folder_with_segs_from_prev_stage'

        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            False,
        )

        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists_or_source_folder,
            seg_from_prev_stage_files,
            output_filename_truncated,
            num_processes_preprocessing,
        )

        for case_idx, preprocessed in enumerate(data_iterator):
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            properties = preprocessed["data_properties"]
            case_id = self._resolve_case_id(preprocessed, case_idx)
            case_id = case_id.replace(os.sep, "_")
            case_dir = join(output_folder, case_id)
            maybe_mkdir_p(case_dir)

            print(f"\nExtracting features for {case_id}:")
            print(f"perform_everything_on_device: "
                  f"{self.perform_everything_on_device}")

            _, features = self.predict_logits_and_features_from_preprocessed_data(
                data=data,
                layer_names=layer_names,
                capture=capture,
                store_all_calls=store_all_calls,
            )

            for layer_name, tensors in features.items():
                safe_layer_name = layer_name.replace(".", "_")
                layer_dir = join(case_dir, safe_layer_name)
                maybe_mkdir_p(layer_dir)

                if len(tensors) == 0:
                    print(f"No tensors captured for layer {layer_name}")
                    continue

                patch_feature_paths = []
                if save_patch_features:
                    for idx, tensor in enumerate(tensors):
                        patch_path = join(layer_dir, f"patch_{idx:04d}.npy")
                        np.save(patch_path, tensor.numpy())
                        patch_feature_paths.append(patch_path)

                # Stack over sliding window calls
                stacked = torch.stack(tensors, dim=0)

                # Aggregate ONLY over patch dimension (keep spatial + channel info)
                aggregated = stacked.mean(dim=0)  # shape: (B, C, X, Y, Z) or (C, X, Y, Z)

                # Remove batch dim if present
                if aggregated.ndim == 5 and aggregated.shape[0] == 1:
                    aggregated = aggregated[0]  # -> (C, X, Y, Z)

                aggregated_path = join(layer_dir, "feature_map.npy")
                np.save(aggregated_path, aggregated.numpy())

                layer_meta = {
                    "layer_name": layer_name,
                    "capture": capture,
                    "num_calls": len(tensors),
                    "feature_map_path": aggregated_path,
                    "feature_map_shape": list(aggregated.shape),
                    "description": "Aggregated full feature map (C, X, Y, Z)",
                    "save_patch_features": save_patch_features,
                    "patch_feature_paths": patch_feature_paths,
                    "first_tensor_shape": list(tensors[0].shape),
                }
                with open(join(layer_dir, "meta.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(layer_meta, f, indent=2)

            with open(join(case_dir, "case_properties.json"), "w",
                      encoding="utf-8") as f:
                json.dump(recursive_fix_for_json_export(properties), f, indent=2)

            print(f"Saved features for {case_id} to {case_dir}")

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    parser.add_argument(
        '--layers',
        nargs='+',
        required=True,
        help='One or more exact layer names from network.named_modules() '
             'for feature extraction.',
    )
    parser.add_argument(
        '--feature_output_dir',
        type=str,
        required=True,
        help='Directory where extracted features will be saved.',
    )
    parser.add_argument(
        '--capture',
        type=str,
        default='output',
        choices=['input', 'output'],
        help='Whether to capture the input or output tensor of the '
             'specified layer(s). Default: output',
    )
    parser.add_argument(
        '--store_all_calls',
        action='store_true',
        default=False,
        help='Store all hooked calls, for example one tensor per '
             'sliding-window patch. By default only the most recent call '
             'is kept before aggregation.',
    )
    parser.add_argument(
        '--save_patch_features',
        action='store_true',
        default=False,
        help='Also save every captured patch tensor as an individual .npy '
             'file. This can use a lot of disk space.',
    )

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    extractor = nnUNetFeatureExtractor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=False,
                                allow_tqdm=not args.disable_progress_bar)
    extractor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    # (Default layer resolution logic removed)

    # Validate layer names
    available_layers = dict(extractor.network.named_modules())
    for layer in args.layers:
        if layer not in available_layers:
            raise ValueError(f"Layer '{layer}' not found in model. Available layers can be printed via named_modules().")

    # Print layer names
    # for name, module in extractor.network.named_modules():
    #     print(name, "->", module.__class__.__name__)

    extractor.extract_features_from_files(
        list_of_lists_or_source_folder=args.i,
        output_folder=args.feature_output_dir,
        layer_names=args.layers,
        capture=args.capture,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
        store_all_calls=args.store_all_calls,
        save_patch_features=args.save_patch_features,
    )


if __name__ == '__main__':
    main()
