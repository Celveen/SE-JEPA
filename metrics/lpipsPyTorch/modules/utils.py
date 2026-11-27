import os
from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # The LPIPS linear-layer weights (v{version}/{net_type}.pth of the official LPIPS release) are
    # read from a local directory given by the LPIPS_WEIGHTS_DIR environment variable. They are only
    # needed when training with training.eval_lpips=true.
    weights_dir = os.environ.get('LPIPS_WEIGHTS_DIR')
    if weights_dir is None:
        raise FileNotFoundError(
            'LPIPS weights not found: set LPIPS_WEIGHTS_DIR to a directory containing '
            f'{net_type}.pth (LPIPS v{version}), or train with training.eval_lpips=false.')
    old_state_dict = torch.load(
        os.path.join(weights_dir, f'{net_type}.pth'),
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
