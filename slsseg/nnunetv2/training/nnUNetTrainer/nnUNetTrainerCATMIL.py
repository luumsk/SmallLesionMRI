"""nnUNetTrainerCATMIL: implemented in the comploss package (comploss.integrations.nnunet).

Install it once in the training environment: pip install -r requirements.txt (pins comploss).
comploss reproduces the original implementation of the paper (its tests/test_parity.py and
tests/test_nnunet.py); the original code is in the git history of this repository (tag arxiv-v5).
"""
from comploss.integrations.nnunet import nnUNetTrainerCATMIL  # noqa: F401
