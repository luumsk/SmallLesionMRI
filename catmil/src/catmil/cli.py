"""Command line: ``catmil losses | profile <labels_dir> | bench``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_mask(path: Path) -> np.ndarray:
    name = path.name
    if name.endswith((".nii.gz", ".nii")):
        try:
            import nibabel as nib
        except ImportError as e:  # pragma: no cover
            raise SystemExit("reading NIfTI needs nibabel: pip install 'catmil[io]'") from e
        return np.asarray(nib.load(str(path)).dataobj) > 0
    if name.endswith(".npy"):
        return np.load(path) > 0
    if name.endswith(".npz"):
        with np.load(path) as z:
            return z[z.files[0]] > 0
    raise ValueError(f"unsupported file {path}")


def cmd_profile(args) -> int:
    from .metrics import lesion_size_profile
    files = sorted(p for p in Path(args.labels_dir).iterdir()
                   if p.name.endswith((".nii.gz", ".nii", ".npy", ".npz")))
    if not files:
        print(f"no label files in {args.labels_dir}", file=sys.stderr)
        return 1
    prof = lesion_size_profile([_load_mask(p) for p in files], connectivity=args.connectivity)
    if args.json:
        print(json.dumps(prof, indent=2))
        return 0
    print(f"{int(prof['cases'])} cases, {int(prof['lesions'])} lesions, "
          f"median size {prof['median_size_vox']:.0f} voxels")
    for b in ("tiny", "small", "medium", "large"):
        print(f"  {b:<7}{int(prof[f'n_{b}']):>7}  {100 * prof[f'frac_{b}']:5.1f}%")
    small = prof["frac_tiny"] + prof["frac_small"]
    print(f"lesions <= 50 voxels: {100 * small:.1f}%"
          + ("  -> many small lesions; CATMIL is designed for this case." if small >= 0.3 else ""))
    return 0


def cmd_losses(_args) -> int:
    from .losses import list_losses
    print("\n".join(list_losses()))
    return 0


def cmd_bench(args) -> int:
    import torch
    from .losses import get_loss
    torch.manual_seed(0)
    b, c, *sp = args.shape
    dev = torch.device(args.device)
    logits = torch.randn(b, c, *sp, device=dev, requires_grad=True)
    # random sparse "lesions": blobs of a few voxels
    tgt = (torch.rand(b, 1, *sp, device=dev) > 0.999).float()
    tgt = torch.nn.functional.max_pool3d(tgt, 3, 1, 1) if len(sp) == 3 else torch.nn.functional.max_pool2d(tgt, 3, 1, 1)
    names = args.losses or ["dice_ce", "tversky", "focal_tversky", "cc_dice_ce", "cat", "mil", "catmil"]
    print(f"input {tuple(logits.shape)} on {dev}, {args.iters} iterations (forward + backward)")
    for name in names:
        kw = {"cat_warmup_epochs": 0} if name == "catmil" else {}
        loss_fn = get_loss(name, **kw)
        for i in range(args.warmup + args.iters):
            if i == args.warmup:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            logits.grad = None
            loss_fn(logits, tgt).backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        print(f"  {name:<16}{1000 * (time.perf_counter() - t0) / args.iters:9.1f} ms/iter")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="catmil", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="lesion-size distribution of a folder of GT masks")
    p.add_argument("labels_dir")
    p.add_argument("--connectivity", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profile)
    p = sub.add_parser("losses", help="list loss names for get_loss()")
    p.set_defaults(func=cmd_losses)
    p = sub.add_parser("bench", help="time forward + backward of each loss")
    p.add_argument("--shape", type=int, nargs="+", default=[2, 2, 64, 64, 64], help="B C *spatial")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--losses", nargs="*")
    p.set_defaults(func=cmd_bench)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
