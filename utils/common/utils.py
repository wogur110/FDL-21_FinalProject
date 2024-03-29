import numpy as np
import torch
import random

def seed_fix(n):
    """
    Fix all random seeds with number n.
    """
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)
    random.seed(n)