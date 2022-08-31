import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

n = 5

tensor_list = [torch.randn(5, 5), torch.zeros(3, 3), torch.randn(4, 4)]
all_sorted, all_sorted_idx = torch.sort(torch.cat([t.view(-1) for t in tensor_list]))

cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in tensor_list]), dim=0)
cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])

split_indeces_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
split_indeces_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
split_indeces = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
                 zip(split_indeces_lt, split_indeces_ge, cum_num_elements[:-1])]

n_smallest = [t.view(-1)[idx] for t, idx in zip(tensor_list, split_indeces)]

# x = np.linspace(0, 10, 10)
# plt.plot(x, x**2)
# plt.xlabel('x')
# plt.ylabel('y')
#
# # displaying the title
# plt.title("Linear graph")
#
# plt.show()

a = torch.tensor([1, 1, 5])
b = torch.tensor([0, 0, 5])
d = torch.sum(a == 0)
print(d)
print(torch.numel(a))

