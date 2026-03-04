from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _safe_squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """Normalize nnUNet targets to integer label maps.

    nnUNet targets are typically shaped (B, 1, *spatial). This helper converts
    them to (B, *spatial) and ensures dtype is `torch.long`.

    Parameters
    ----------
    target:
        Target tensor of shape (B, 1, *spatial) or (B, *spatial).

    Returns
    -------
    torch.Tensor
        Label tensor of shape (B, *spatial) with dtype `torch.long`.
    """
    if target.ndim >= 2 and target.shape[1] == 1:
        return target[:, 0].long()
    return target.long()


def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Compute a foreground probability map from network logits.

    - If `logits` has one channel (C=1): assumes sigmoid-style binary training
      and returns `sigmoid(logits)`.
    - If `logits` has multiple channels (C>=2): assumes softmax-style training
      and returns the probability of "any foreground", computed as the union of
      all non-background classes: sum_{c=1..C-1} softmax(logits)[c].

    Parameters
    ----------
    logits:
        Network output logits of shape (B, C, *spatial).

    Returns
    -------
    torch.Tensor
        Foreground probability of shape (B, 1, *spatial).
    """
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)

    probs = torch.softmax(logits, dim=1)
    fg = probs[:, 1:].sum(dim=1, keepdim=True)
    return fg


class ComponentAdaptiveTverskyLoss(nn.Module):
    """Component-adaptive (lesion-balanced) Tversky loss for binary targets.

    This loss reweights **foreground voxels** inversely by the size of their
    connected component in the *ground-truth* mask. Intuition: each lesion
    (connected component) should contribute more equally to the objective,
    preventing large lesions from dominating the gradients.

    Implementation notes
    --------------------
    - Expects a binary foreground vs background target (labels > 0 are treated
      as foreground).
    - Works with nnUNet softmax outputs by converting logits to a single
      foreground-union probability map.
    - Requires SciPy (`scipy.ndimage`) for connected-component labeling.

    Parameters
    ----------
    alpha, beta:
        Tversky trade-off parameters (FP vs FN).
    gamma:
        Component size exponent. Larger values put more emphasis on small
        lesions.
    eps_cc:
        Small constant added to component size to stabilize very tiny lesions.
    w_bg:
        Background voxel weight (kept small to avoid background dominating).
    smooth:
        Numerical stability constant.
    connectivity:
        Connectivity for connected components (1 is 6-neighborhood in 3D).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        smooth: float = 1e-5,
        connectivity: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps_cc = float(eps_cc)
        self.w_bg = float(w_bg)
        self.smooth = float(smooth)
        self.connectivity = int(connectivity)

    def _weight_map(self, gt_fg: torch.Tensor) -> torch.Tensor:
        """Build per-voxel weights based on GT connected components."""
        try:
            import scipy.ndimage as ndi
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for ComponentAdaptiveTverskyLoss. "
                "Please install scipy or implement a torch CC routine."
            ) from e

        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)
        batch_weights = []

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]

        for b in range(gt_cpu.shape[0]):
            struct = ndi.generate_binary_structure(
                gt_cpu[b].ndim,
                self.connectivity,
            )
            lab, _ = ndi.label(gt_cpu[b], structure=struct)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0

            denom = (sizes + self.eps_cc).astype(np.float32)
            inv = np.zeros_like(denom, dtype=np.float32)
            nonzero = denom > 0
            inv[nonzero] = denom[nonzero] ** (-self.gamma)

            w_fg = inv[lab].astype(np.float32)
            w = np.where(gt_cpu[b] > 0, w_fg, self.w_bg).astype(np.float32)
            batch_weights.append(w)

        w_np = np.stack(batch_weights, axis=0)
        w_t = torch.from_numpy(w_np).to(gt_fg.device)
        return w_t

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        fg_prob = _foreground_probability(net_output)
        tgt = _safe_squeeze_target(target)
        gt_fg = (tgt > 0).float()

        if gt_fg.sum() == 0:
            # No foreground in this patch: avoid unstable CC labeling and
            # return a zero loss contribution.
            return fg_prob.new_tensor(0.0)

        w = self._weight_map(gt_fg).unsqueeze(1)

        spatial_dims = list(range(2, fg_prob.ndim))
        gt = gt_fg.unsqueeze(1)
        tp = (w * fg_prob * gt).sum(dim=spatial_dims)
        fp = (w * fg_prob * (1.0 - gt)).sum(dim=spatial_dims)
        fn = (w * (1.0 - fg_prob) * gt).sum(dim=spatial_dims)

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()


class ComponentAdaptiveTverskyLossMultiClass(nn.Module):
    """Component-adaptive Tversky loss for multi-class segmentation.

    This variant applies a component-adaptive weighting **per foreground class**
    (excluding background class 0) and averages the loss across the classes
    that are present in the current patch.

    Notes
    -----
    - Expects softmax-style multi-class logits (C>=2). For binary (C=1), it
      falls back to a single foreground channel.
    - Component weights are computed from *ground-truth* connected components
      **within each class mask**.
    - Uses SciPy on CPU for connected-component labeling.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        smooth: float = 1e-5,
        connectivity: int = 1,
        class_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps_cc = float(eps_cc)
        self.w_bg = float(w_bg)
        self.smooth = float(smooth)
        self.connectivity = int(connectivity)
        self.class_ids = class_ids

    def _weight_map(self, gt_fg: torch.Tensor) -> torch.Tensor:
        """Build per-voxel weights based on GT connected components."""
        try:
            import scipy.ndimage as ndi
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for ComponentAdaptiveTverskyLossMultiClass. "
                "Please install scipy or implement a torch CC routine."
            ) from e

        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)
        batch_weights = []

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]

        for b in range(gt_cpu.shape[0]):
            struct = ndi.generate_binary_structure(
                gt_cpu[b].ndim,
                self.connectivity,
            )
            lab, _ = ndi.label(gt_cpu[b], structure=struct)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0

            denom = (sizes + self.eps_cc).astype(np.float32)
            inv = np.zeros_like(denom, dtype=np.float32)
            nonzero = denom > 0
            inv[nonzero] = denom[nonzero] ** (-self.gamma)

            w_fg = inv[lab].astype(np.float32)
            w = np.where(gt_cpu[b] > 0, w_fg, self.w_bg).astype(np.float32)
            batch_weights.append(w)

        w_np = np.stack(batch_weights, axis=0)
        w_t = torch.from_numpy(w_np).to(gt_fg.device)
        return w_t

    def _tversky_from_prob(
        self,
        prob: torch.Tensor,
        gt_fg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute component-adaptive Tversky from a prob map and fg mask."""
        if gt_fg.sum() == 0:
            return prob.new_tensor(0.0)

        w = self._weight_map(gt_fg).unsqueeze(1)
        gt = gt_fg.unsqueeze(1)
        spatial_dims = list(range(2, prob.ndim))

        tp = (w * prob * gt).sum(dim=spatial_dims)
        fp = (w * prob * (1.0 - gt)).sum(dim=spatial_dims)
        fn = (w * (1.0 - prob) * gt).sum(dim=spatial_dims)

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        tgt = _safe_squeeze_target(target)

        if net_output.shape[1] == 1:
            prob = torch.sigmoid(net_output)
            gt_fg = (tgt > 0).float()
            return self._tversky_from_prob(prob, gt_fg)

        probs = torch.softmax(net_output, dim=1)
        n_classes = int(probs.shape[1])

        if self.class_ids is None:
            class_ids = list(range(1, n_classes))
        else:
            class_ids = [
                int(c)
                for c in self.class_ids
                if 0 <= int(c) < n_classes
            ]

        losses: list[torch.Tensor] = []
        for c in class_ids:
            gt_c = (tgt == c).float()
            if gt_c.sum() == 0:
                continue
            prob_c = probs[:, c : c + 1]
            losses.append(self._tversky_from_prob(prob_c, gt_c))

        if len(losses) == 0:
            return probs.new_tensor(0.0)

        return torch.stack(losses).mean()


class nnUNetTrainerCAT(nnUNetTrainer):
    """nnUNet trainer variant using component-adaptive Tversky (CAT).

    This trainer extends the default nnUNet training objective by adding a
    **Component-Adaptive Tversky (CAT)** loss term that balances gradients
    across connected components (lesions) in the ground-truth mask.

    Automatic binary / multi-class behavior
    ---------------------------------------
    The trainer selects the appropriate CAT formulation per batch:

    - **Binary targets** (network output has C=1 or C=2 channels):
      :class:`ComponentAdaptiveTverskyLoss` is used. Foreground is treated as
      the union of all non-background labels.

    - **Multi-class targets** (network output has C>2 channels, e.g. BraTS
      metastasis C=4):
      :class:`ComponentAdaptiveTverskyLossMultiClass` is used. CAT is applied
      independently per foreground class and averaged across classes present
      in the patch.

    In both cases, the CAT term is added to the standard nnUNet loss:

        L = L_base + λ_cat * L_CAT

    where ``L_base`` is the default nnUNet loss (Dice + Cross Entropy with
    deep supervision) and ``λ_cat`` is linearly warmed up during the first
    training epochs.

    Implementation notes
    --------------------
    - Connected components are computed from the **ground-truth mask** using
      ``scipy.ndimage``.
    - CAT is applied only to the **highest-resolution prediction**
      (``output[0]``) to avoid expensive component computations on deep
      supervision outputs.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda'),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )

        # CAT hyperparameters
        alpha = 0.3
        beta = 0.7
        gamma = 1.0
        eps_cc = 5.0
        w_bg = 0.01
        connectivity = 1
        lambda_cat = 0.1

        self._cat_params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'eps_cc': float(eps_cc),
            'w_bg': float(w_bg),
            'smooth': 1e-5,
            'connectivity': int(connectivity),
        }

        # If cat_class_ids is None, no specific class list is provided, so CAT-MC automatically applies
        # to all foreground classes (1..C-1). For binary segmentation (C=1 or C=2),
        # this parameter is ignored because CAT binary is used instead of CAT-MC.
        self.cat_class_ids: list[int] | None = None
        self._cat_params_mc = dict(self._cat_params)

        self.lambda_cat_final = float(lambda_cat)
        self.lambda_cat_warmup_epochs = 50
        self.lambda_cat = 0.0

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        warmup = int(self.lambda_cat_warmup_epochs)
        if warmup <= 0:
            self.lambda_cat = float(self.lambda_cat_final)
            return

        # nnUNet uses 0-based epochs.
        t = min(max(self.current_epoch, 0), warmup) / float(warmup)
        self.lambda_cat = float(t * self.lambda_cat_final)

    def _build_loss(self) -> nn.Module:
        base = super()._build_loss()
        cat_bin = ComponentAdaptiveTverskyLoss(**self._cat_params)
        cat_mc = ComponentAdaptiveTverskyLossMultiClass(
            **self._cat_params_mc,
            class_ids=self.cat_class_ids,
        )

        class _CATWrappedLoss(nn.Module):
            def __init__(
                self,
                base_loss: nn.Module,
                cat_bin_loss: nn.Module,
                cat_mc_loss: nn.Module,
                trainer_ref: nnUNetTrainer,
            ) -> None:
                super().__init__()
                self.base_loss = base_loss
                self.cat_bin_loss = cat_bin_loss
                self.cat_mc_loss = cat_mc_loss
                self.trainer = trainer_ref
                self._logged_mode = True

            def forward(self, net_output, target) -> torch.Tensor:
                if isinstance(net_output, (list, tuple)):
                    out = net_output[0]
                else:
                    out = net_output

                if isinstance(target, (list, tuple)):
                    tgt = target[0]
                else:
                    tgt = target

                # Prefer detecting multi-class by model output channels.
                # This is robust even if labels are not {0,1} (e.g. {0,2}).
                n_ch = int(out.shape[1])
                is_multiclass = n_ch > 2

                # Log once which CAT mode is used (binary or multi-class)
                if self._logged_mode:
                    mode = "CAT-MC (multi-class)" if is_multiclass else "CAT-Binary"
                    print(f"[nnUNetTrainerCAT] Using {mode}, channels={n_ch}")
                    self._logged_mode = False

                base_term = self.base_loss(net_output, target)
                lambda_cat = float(self.trainer.lambda_cat)

                # During warmup lambda_cat can be 0.0. Avoid computing CAT in
                # that case because 0.0 * NaN is still NaN in PyTorch.
                if lambda_cat <= 0.0:
                    return base_term

                if is_multiclass:
                    cat_term = self.cat_mc_loss(out, tgt)
                else:
                    cat_term = self.cat_bin_loss(out, tgt)

                # Extra safety: ensure CAT cannot introduce NaNs/Infs.
                cat_term = torch.nan_to_num(
                    cat_term,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                return base_term + lambda_cat * cat_term

        return _CATWrappedLoss(base, cat_bin, cat_mc, self)
