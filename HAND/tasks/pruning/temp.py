import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import os

# n = 5
#
# tensor_list = [torch.randn(5, 5), torch.zeros(3, 3), torch.randn(4, 4)]
# all_sorted, all_sorted_idx = torch.sort(torch.cat([t.view(-1) for t in tensor_list]))
#
# cum_num_elements = torch.cumsum(torch.tensor([t.numel() for t in tensor_list]), dim=0)
# cum_num_elements = torch.cat([torch.tensor([0]), cum_num_elements])
#
# split_indeces_lt = [all_sorted_idx[:n] < cum_num_elements[i + 1] for i, _ in enumerate(cum_num_elements[1:])]
# split_indeces_ge = [all_sorted_idx[:n] >= cum_num_elements[i] for i, _ in enumerate(cum_num_elements[:-1])]
# split_indeces = [all_sorted_idx[:n][torch.logical_and(lt, ge)] - c for lt, ge, c in
#                  zip(split_indeces_lt, split_indeces_ge, cum_num_elements[:-1])]
#
# n_smallest = [t.view(-1)[idx] for t, idx in zip(tensor_list, split_indeces)]
#
# # x = np.linspace(0, 10, 10)
# # y = x**2
# # z = x**3
#
# plt.plot(x, x, label = "x")
# plt.plot(x, y, label = "x^2")
# plt.plot(x, z, label = "x^3")
# plt.legend()
# plt.xlabel('x')
# # displaying the title
# plt.title("Linear graph")
# plt.show()

import json


def write_list(a_list):
    print("Started writing list data into a json file")
    with open("names.json", "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('names.json', 'rb') as fp:
        n_list = json.load(fp)
        return n_list


# assume you have the following list
names = ['Jessa', 'Eric', 'Bob']
write_list(names)
r_names = read_list()
print('List is', r_names)
