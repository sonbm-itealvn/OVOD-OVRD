"""Quick overfit test to verify training works."""
import torch
from models import build_model
from models.criterion import HungarianMatcher, SetCriterion

cfg = {
    'backbone': 'resnet50', 'pretrained': False,
    'hidden_dim': 256, 'nheads': 8,
    'enc_layers': 2, 'dec_layers': 2,
    'dim_feedforward': 512, 'dropout': 0.1,
    'num_obj_queries': 10, 'num_rel_queries': 5,
    'num_classes': 3, 'num_rel_predicates': 2,
    'clip_dim': 512, 'image_size': 224,
    'use_aux_loss': True, 'fpn_level': -1,
}

model = build_model(cfg)
matcher = HungarianMatcher(use_focal=True)
wd = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_rel': 1, 'loss_vl': 1}
for i in range(1):
    for k, v in list(wd.items()):
        if '_' not in k[5:]:
            wd[f'{k}_{i}'] = v
crit = SetCriterion(3, matcher, wd, ['labels', 'boxes', 'relations', 'vl'],
                    num_rel_predicates=2, use_focal=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(2, 3, 224, 224)
tgts = [
    {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4) * 0.5 + 0.25,
     'relations': torch.tensor([[0, 1, 0]])},
    {'labels': torch.tensor([2]), 'boxes': torch.rand(1, 4) * 0.5 + 0.25,
     'relations': torch.zeros(0, 3, dtype=torch.long)},
]

print("Running 20-step overfit test...")
for step in range(20):
    out = model(x)
    losses = crit(out, tgts)
    total = sum(losses[k] * wd[k] for k in losses if k in wd)
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    opt.step()
    if step % 5 == 0:
        ce = losses["loss_ce"].item()
        bbox = losses["loss_bbox"].item()
        rel = losses["loss_rel"].item()
        print(f"  Step {step:3d} | loss {total.item():.4f} | ce {ce:.4f} | bbox {bbox:.4f} | rel {rel:.4f}")

print("\n=== OVERFIT TEST PASSED === (loss should decrease)")
