import torch

SAM2_CONFIG = {
    'model': 'sam2_hiera_l.yaml',
    'checkpoint': './checkpoints/sam2_hiera_large.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'multimask_output': True,
    'points_per_side': 32  # per auto-segmentazione
}