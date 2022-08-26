from typing import List
import numpy as np
from tqdm import tqdm
import numpy as np

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
# path_distance = lambda r, c:
# path_distance = lambda r, w: get_pair_cosine_sim() for weight in w[r]
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate((r[0:i], r[k:-len(r) + i - 1:-1], r[k + 1:len(r)]))


def path_distance(route, weights):
    permuted_weights = weights[route]
    normed_weights = permuted_weights / np.expand_dims(np.linalg.norm(permuted_weights, axis=-1), -1)
    return 1 - (normed_weights[:-1] * normed_weights[1:]).mean()


def two_opt(weights, improvement_threshold):  # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(weights.shape[0])  # Make an array of row numbers corresponding to cities.
    improvement_factor = 1  # Initialize the improvement factor.
    best_distance = path_distance(route, weights)  # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold:  # If the route is still improving, keep going!
        # print(improvement_factor)
        distance_to_beat = best_distance  # Record the distance at the beginning of the loop.
        for swap_first in range(1, len(route) - 2):  # From each city except the first and last,
            for swap_last in range(swap_first + 1, len(route)):  # to each of the cities following,
                new_route = two_opt_swap(route, swap_first, swap_last)  # try reversing the order of these cities
                new_distance = path_distance(new_route, weights)  # and check the total distance with this modification.
                if new_distance < best_distance:  # If the path distance is an improvement,
                    route = new_route  # make this the accepted best route
                    best_distance = new_distance  # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance / distance_to_beat  # Calculate how much the route has improved.
    return route  # When the route is no longer improving substantially, stop searching and return the route.


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
    max_sims_order = [0]
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


def get_pair_cosine_sim(source: np.array, target: np.array) -> np.array:
    source_normed = source / np.expand_dims(np.linalg.norm(source), -1)
    target_normed = target / np.expand_dims(np.linalg.norm(target), -1)
    return np.dot(source_normed, target_normed.T)


def get_one_cosine_sim(source_weights: np.array, weights: np.array) -> np.array:
    normed_weights = weights / np.expand_dims(np.linalg.norm(weights, axis=-1), -1)
    source_normed_weights = source_weights / np.expand_dims(np.linalg.norm(source_weights), -1)
    return np.dot(source_normed_weights, normed_weights.T)


def get_all_cosine_sim(weights: np.array) -> np.array:
    if weights.shape[1] == 1:
        normalized_weights = weights / np.linalg.norm(weights, keepdims=True)
        return -(normalized_weights-normalized_weights.T)**2
    normed_weights = weights / np.expand_dims(np.linalg.norm(weights, axis=-1), -1)
    normed_weights = normed_weights.reshape(normed_weights.shape[0], -1)
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
    weights = np.random.normal(0, 1, (500, 9))
    # weights = np.array([[1, 2, 3], [3, 6, 12.1], [2, 4, 6], [3, 6, 12.2], [3, 6, 12.1]])
    # weights = np.array([[1, 2, 3], [3, 6, 12.9], [1, 4, 6], [4, 6, 12.2], [3, 6, 12.1]])
    max_sims_order = get_max_sim_order(weights, False)
    two_opt_order = two_opt(weights, 0.001)
    # print(max_sims_order)
    print(f"Prev tot sim: {get_tot_sim(get_all_cosine_sim(weights), list(range(1, len(weights))))}")
    print(f"two opt tot sim: {get_tot_sim(get_all_cosine_sim(weights), two_opt_order)}")
    print(f"max_sim tot sim: {get_tot_sim(get_all_cosine_sim(weights), max_sims_order)}")
