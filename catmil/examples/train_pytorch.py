"""Minimal plain-PyTorch training loop with CATMIL and per-term logging (synthetic data).

python examples/train_pytorch.py
"""
import torch
import torch.nn as nn

from catmil import CATMILLoss, LabelComponents
from catmil.metrics import lesion_detection_metrics

torch.manual_seed(0)


def synthetic_batch(b=2, size=32):
    """Images with a few bright 3x3x3 'lesions' and their labels."""
    label = torch.zeros(b, 1, size, size, size)
    for i in range(b):
        for _ in range(4):
            z, y, x = torch.randint(2, size - 3, (3,))
            label[i, 0, z:z + 3, y:y + 3, x:x + 3] = 1
    image = torch.randn_like(label) + 2 * label
    return image, label


model = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 2, 1))
opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = CATMILLoss(cat_warmup_epochs=5)
label_cc = LabelComponents()

for epoch in range(10):
    loss_fn.set_epoch(epoch)
    image, label = synthetic_batch()
    components = torch.stack([label_cc(l) for l in label])  # normally done in the DataLoader workers
    out = loss_fn(model(image), label, components=components, return_components=True)
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    print(f"epoch {epoch}: loss {out['loss'].item():.3f}  base {out['base']:.3f}  "
          f"cat {out['cat']:.3f}  mil {out['mil']:.3f}  lambda_cat {out['lambda_cat']:.3f}")

image, label = synthetic_batch(1)
pred = model(image).argmax(1)[0].numpy()
print(lesion_detection_metrics(label[0, 0].numpy(), pred))
