import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import os

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
# y = x**2
# z = x**3

plt.plot(x, x, label = "x")
plt.plot(x, y, label = "x^2")
plt.plot(x, z, label = "x^3")
plt.legend()
plt.xlabel('x')
# displaying the title
plt.title("Linear graph")
plt.show()

def write_list(a_list):
    here = os.path.dirname(os.path.abspath(__file__))
    # store list in binary file so 'wb' mode
    with open(os.path.join(here, "state.pickle"), "wb") as fp:
        pickle.dump(names, fp)
        print('Done writing list into a binary file')


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('relative_recon_accuracies.txt', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


# list of names
names = ['Jessa', 'Eric', 'Bob']
write_list(names)
# r_names = read_list()
# print('List is', r_names)
