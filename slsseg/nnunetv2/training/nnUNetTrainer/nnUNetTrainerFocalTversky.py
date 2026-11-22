"""nnUNetTrainerFocalTversky: implemented in the catmil package (catmil/src/catmil/integrations/nnunet.py).

Install it once in the training environment: pip install -e <repo>/catmil
The losses there reproduce the original implementation (catmil/tests/test_parity.py,
catmil/tests/test_nnunet.py); the original code is in git history before this change.
"""
from catmil.integrations.nnunet import nnUNetTrainerFocalTversky  # noqa: F401
