from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import nn


class FeatureHookManager:
    """Register forward hooks by layer name and store extracted features.

    By default, this stores the output tensor of each hooked layer. For layers
    such as segmentation heads, `capture="input"` can be used to store the
    tensor entering the layer instead.

    Parameters
    ----------
    model:
        PyTorch model whose named modules will be searched.
    layer_names:
        Exact names from `model.named_modules()`.
    capture:
        One of {"output", "input"}.
    detach:
        Whether to detach stored tensors from the graph.
    move_to_cpu:
        Whether to move stored tensors to CPU before saving.
    store_all_calls:
        If True, store a list of tensors for repeated calls. This is useful for
        sliding-window inference, where a hooked layer is called once per patch.
        If False, only the most recent tensor is kept.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        capture: str = "output",
        detach: bool = True,
        move_to_cpu: bool = True,
        store_all_calls: bool = True,
    ) -> None:
        if capture not in {"output", "input"}:
            raise ValueError(
                f"capture must be 'output' or 'input'. Got: {capture}"
            )

        self.model = model
        self.layer_names = layer_names
        self.capture = capture
        self.detach = detach
        self.move_to_cpu = move_to_cpu
        self.store_all_calls = store_all_calls

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[str, List[torch.Tensor]] = defaultdict(list)

        available = dict(model.named_modules())
        missing = [name for name in layer_names if name not in available]
        if missing:
            raise ValueError(
                "These layer names were not found in the model: "
                f"{missing}"
            )

        self._module_lookup = available

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach:
            tensor = tensor.detach()
        if self.move_to_cpu:
            tensor = tensor.cpu()
        return tensor

    def _make_hook(self, layer_name: str):
        def hook(
            module: nn.Module,
            inputs,
            output,
        ) -> None:
            if self.capture == "input":
                if not inputs:
                    raise RuntimeError(
                        f"Layer '{layer_name}' did not receive inputs."
                    )
                value = inputs[0]
            else:
                value = output

            if isinstance(value, (tuple, list)):
                if len(value) == 0:
                    raise RuntimeError(
                        f"Layer '{layer_name}' returned an empty sequence."
                    )
                value = value[0]

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Hooked value for layer '{layer_name}' is not a tensor. "
                    f"Got type: {type(value)}"
                )

            value = self._prepare_tensor(value)

            if self.store_all_calls:
                self.features[layer_name].append(value)
            else:
                self.features[layer_name] = [value]

        return hook

    def register(self) -> None:
        """Register hooks on all requested layers."""
        self.remove()
        self.clear()

        for layer_name in self.layer_names:
            module = self._module_lookup[layer_name]
            handle = module.register_forward_hook(
                self._make_hook(layer_name)
            )
            self.handles.append(handle)

    def clear(self) -> None:
        """Clear stored features."""
        self.features = defaultdict(list)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_features(self) -> Dict[str, List[torch.Tensor]]:
        """Return stored features."""
        return dict(self.features)

    def get_last(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return the most recent stored tensor for one layer."""
        values = self.features.get(layer_name, [])
        return values[-1] if values else None