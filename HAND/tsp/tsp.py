from typing import List
import numpy as np
from tqdm import tqdm


def get_max_sim_order(weights: np.array, calc_all=True) -> List[int]:
    """
    Args:
        weights: numpy array of N weights, each weight is of K length (NXK)

    Returns:
        A list of indices, specifying the order of the maximum similarity order (N-1)
    """
    n = len(weights)
    if calc_all:
        all_sims = get_all_cosine_sim(weights)
    max_sims_order = []
    possible_indices = [i for i in range(1, n)]
    curr_index = 0
    for _ in tqdm(range(n - 1)):
        curr_possible_indices = possible_indices.copy()
        # Remove the current index from the current possible indices
        if curr_index in curr_possible_indices:
            curr_possible_indices.pop(curr_possible_indices.index(curr_index))
        if calc_all:
            possible_values = all_sims[curr_index, curr_possible_indices]
        else:
            possible_values = get_one_cosine_sim(weights[curr_index], weights[curr_possible_indices])

        min_index_in_possible = np.argmax(possible_values).item()
        real_min_index = curr_possible_indices[min_index_in_possible]
        max_sims_order.append(real_min_index)
        possible_indices.pop(possible_indices.index(real_min_index))
        # Jump to the min index
        curr_index = real_min_index
    return max_sims_order


def get_one_cosine_sim(source_weights: np.array, weights: np.array) -> np.array:
    normed_weights = weights / np.expand_dims(np.linalg.norm(weights, axis=-1), -1)
    source_normed_weights = source_weights / np.expand_dims(np.linalg.norm(source_weights), -1)
    return np.dot(source_normed_weights, normed_weights.T)


def get_all_cosine_sim(weights: np.array) -> np.array:
    normed_weights = weights / np.expand_dims(np.linalg.norm(weights, axis=-1), -1)
    return np.dot(normed_weights, normed_weights.T)


def get_tot_sim(sim_matrix, order):
    n = len(sim_matrix)
    sum = 0.0
    curr_index = 0
    for i in range(n - 1):
        sum += sim_matrix[curr_index, order[i]]
        curr_index = order[i]
    return sum


if __name__ == '__main__':
    # weights = np.random.normal(0, 1, (10000, 9))
    weights = np.array([[1, 2, 3], [3, 6, 12.1], [2, 4, 6], [3, 6, 12.2], [3, 6, 12.1]])
    # weights = np.array([[1, 2, 3], [3, 6, 12.9], [1, 4, 6], [4, 6, 12.2], [3, 6, 12.1]])
    max_sims_order = get_max_sim_order(weights, False)
    print(f"Prev tot sim: {get_tot_sim(get_all_cosine_sim(weights), list(range(1, len(weights))))}")
    print(f"Curr tot sim: {get_tot_sim(get_all_cosine_sim(weights), max_sims_order)}")
