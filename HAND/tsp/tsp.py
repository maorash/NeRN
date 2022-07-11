from typing import List

import torch


def get_min_diff_order(weights: torch.Tensor):
    all_diffs = get_all_diffs(weights)


def get_all_diffs(weights: torch.Tensor) -> List[int]:


if __name__ == '__main__':
    weights = torch.normal(0, 1, (20, 3, 3))
    get_min_diff_order(weights)
