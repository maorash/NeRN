from typing import List
import numpy as np
from tqdm import tqdm


def get_min_diff_order(weights: np.array) -> List[int]:
    n = len(weights)
    all_diffs = get_all_cossine_sim(weights)
    min_diffs_order = []
    possible_indices = [i for i in range(1, n)]
    curr_index = 0
    for row_num in tqdm(range(n - 1)):
        curr_possible_indices = possible_indices.copy()
        if curr_index in curr_possible_indices:
            curr_possible_indices.pop(curr_possible_indices.index(curr_index))
        possible_values = all_diffs[curr_index, curr_possible_indices]
        curr_min_index = np.argmax(possible_values).item()
        real_index = curr_possible_indices[curr_min_index]
        min_diffs_order.append(real_index)
        possible_indices.pop(possible_indices.index(real_index))
        curr_index = real_index
    return min_diffs_order


def get_all_cossine_sim(weights: np.array) -> np.array:
    normed_weights = weights / np.expand_dims(np.linalg.norm(weights, axis=-1), -1)
    return np.dot(normed_weights, normed_weights.T)


def get_tot_sim(sim_matrix, order):
    n = len(sim_matrix)
    sum = 0.0
    curr_index=0
    for i in range(n - 1):
        # print(sim_matrix[i, order[i]])
        sum += sim_matrix[curr_index, order[i]]
        curr_index = order[i]
    return sum


if __name__ == '__main__':
    weights = np.random.normal(0, 1, (1000, 9))
    # weights = np.array([[1, 2, 3], [3, 6, 12.1], [2, 4, 6], [3, 6, 12.2], [3, 6, 12.1]])
    # weights = np.array([[1, 2, 3], [3, 6, 12.9], [1, 4, 6], [4, 6, 12.2], [3, 6, 12.1]])
    max_sims_order = get_min_diff_order(weights)
    print(f"Prev tot sim: {get_tot_sim(get_all_cossine_sim(weights), list(range(1, len(weights))))}")
    print(f"Curr tot sim: {get_tot_sim(get_all_cossine_sim(weights),max_sims_order)}")
