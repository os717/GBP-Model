from numba import njit
import numpy as np
import random
import math
import time
import queue
from typing import Dict, List, Tuple
from numba.typed import Dict
from numba import types
from numba import njit, types
from numba.typed import Dict, List

def split_values_random(N, n, P):

    # create list of all node indices
    values = list(range(N))

    # Shuffle the values randomly
    random.shuffle(values)
    
    # Initialize the result list with P empty lists
    result = [[] for _ in range(P)]
    
    # Distribute the values into the sublists
    for i, value in enumerate(values):
        sublist_index = i % P
        if len(result[sublist_index]) < n:
            result[sublist_index].append(value)
    
    return result

def split_values_fixed(np, k):
    values = list(range(np))
    n = len(values)
    i = 0
    while True:
        # Get the next k values starting from index i and wrapping around
        sub_list = values[i:i+k]
        if len(sub_list) < k:
            remaining = k - len(sub_list)
            sub_list += values[:remaining]
            i = (i + remaining) % n
        else:
            i = (i + k) % n
        yield sub_list

def create_slices(values, n, P):
    slices = []
    for i in range(0, len(values), n):
        if len(slices) == P:
            break
        slices.append(values[i:i+n])
    return slices

def iterate_with_wrap_around(values, k):
    n = len(values)
    i = 0
    while True:
        # Get the next k values starting from index i and wrapping around
        sub_list = values[i:i+k]
        if len(sub_list) < k:
            sub_list += values[:k - len(sub_list)]
        i = (i + k) % n
        yield sub_list

def split_values_fixed(np, k):
    values = list(range(np))
    n = len(values)
    i = 0
    while True:
        # Get the next k values starting from index i and wrapping around
        sub_list = values[i:i+k]
        if len(sub_list) < k:
            remaining = k - len(sub_list)
            sub_list += values[:remaining]
            i = (i + remaining) % n
        else:
            i = (i + k) % n
        yield sub_list

def iterate_with_wrap_around(values, k):
    n = len(values)
    i = 0
    while True:
        # Get the next k values starting from index i and wrapping around
        sub_list = values[i:i+k]
        if len(sub_list) < k:
            sub_list += values[:k - len(sub_list)]
        i = (i + k) % n
        yield sub_list

@njit(fastmath=True)
def l2_dist(mat1: np.array, mat2: np.array) -> np.array:
    """Return the L2 distance between two tensors"""
    return np.power(np.sum(np.power(mat1 - mat2, 2)), 1/2)

@njit(fastmath=True)
def find_neighbors_index(A: np.array) -> list:    
    return [np.where(A[:, i])[0] for i in range(A.shape[0])]

@njit(fastmath=True)
def find_edges(A):
    """Return 2d numpy array with all the edges"""
    node_i, node_j = np.where(A > 0)
    non_diag = np.where(node_i - node_j)[0]
    node_i = node_i[non_diag]
    node_j = node_j[non_diag]
    return np.stack((node_i, node_j), axis=0).T

@njit(fastmath=True)
def find_edges_UPDATED(A, num):
    """Return 2d numpy array with all the edges"""
    edges = []
    for i in range (np.shape(A)[0]):
        if i != num:
            if A[i][num] != 0:
                # print(A[i][num])
                edge = [i,num]
                edges.append(edge)
    return edges

@njit(fastmath=True)
def initialize_m_ij(A, b):
    P_ii = np.diag(A)
    mu_ii = b / P_ii
    P_ij = np.zeros_like(A)
    mu_ij = np.zeros_like(A)
    return P_ii, mu_ii, P_ij, mu_ij

@njit(fastmath=True)
def calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i):
    """Optimized version of the calc_m_ij function."""
    A_ij = A[i][j]

    # Store frequently accessed elements in local variables to minimize memory access
    P_ij_ji = P_ij[j][i]
    mu_ij_ji = mu_ij[j][i]

    # Calculate these arrays once to avoid repeated computation
    P_ij_Ni_i = P_ij[N_i[i], i]
    mu_ij_Ni_i = mu_ij[N_i[i], i]

    # P_i\j = p_ii + ∑_{k ∈ N(i)\j} Pₖᵢ
    P_i_wout_j = P_ii[i] + np.sum(P_ij_Ni_i) - P_ij_ji

    P_ij_ij = -A_ij ** 2 / P_i_wout_j

    # μ_i\j = P_i\j ^ (-1)  * (pᵢᵢμᵢᵢ + ∑_{k ∈ N(i)\j} Pₖᵢμₖᵢ)
    P_mu_ii = P_ii[i] * mu_ii[i]
    P_mu_ij = P_ij_Ni_i * mu_ij_Ni_i

    # Calculate mu_i_wout_j based on whether P_i_wout_j is zero or not
    mu_i_wout_j = (P_mu_ii + np.sum(P_mu_ij) - P_ij_ji * mu_ij_ji) / P_i_wout_j

    # Check if denominator is zero to avoid division by zero
    mu_ij_ij = -A_ij * mu_i_wout_j / P_ij_ij

    return P_ij_ij, mu_ij_ij

@njit(fastmath=True)
def calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i, capping=None):
    """Get the marginal precision and the marginal mean for a given node"""
    # Calculate the sum of precision terms of neighboring nodes
    sum_P_ij_N_i = np.sum(P_ij[N_i[i], i])

    # Calculate the marginal precision Pᵢ = pᵢᵢ + ∑_{k ∈ N(i)} Pₖᵢ
    P_i = P_ii[i] + sum_P_ij_N_i

    # Calculate the marginal mean μᵢ = (pᵢᵢμᵢᵢ + ∑_{k ∈ N(i)} Pₖᵢμₖᵢ) / Pᵢ
    if P_i != 0:  # Check if denominator is zero to avoid division by zero
        P_mu_ii = P_ii[i] * mu_ii[i]
        P_mu_ij = P_ij[N_i[i], i] * mu_ij[N_i[i], i]
        mu_i = (P_mu_ii + np.sum(P_mu_ij)) / P_i
    else:
        mu_i = 0  # Handle it appropriately based on your application logic

    return P_i, mu_i

@njit(fastmath=True)
def assign_values_to_pes_random(num_nodes, node_updates_per_pe, number_pes, mode='random'):
    
    if mode == 'random':
        # Create a pool of values to choose from
        values_pool = np.arange(num_nodes)
        np.random.shuffle(values_pool)
        
        # Distribute values among PEs
        PEs = {}
        for i in range(number_pes):
            start_index = i * node_updates_per_pe
            end_index = min((i + 1) * node_updates_per_pe, num_nodes)
            pe_values = values_pool[start_index:end_index]
            PEs[i] = pe_values

    return PEs

@njit(fastmath=True)
def assign_values_to_pes_fixed(num_nodes, node_update_schedule, node_updates_per_pe, number_pes, current_index, shuffle_fixed=True):
    
    # print(node_update_schedule)

    # Explicitly specify types for all variables
    total_num_nodes = number_pes * node_updates_per_pe  # int
    start_index = current_index  # int
    end_index = (start_index + total_num_nodes) % len(node_update_schedule)  # int

    if end_index > start_index:
        values_pool = node_update_schedule[start_index:end_index]
    else:
        values_pool = np.concatenate((node_update_schedule[start_index:], node_update_schedule[:end_index]))

    PEs = {}
    for i in range(number_pes):
        start_index = i * node_updates_per_pe
        end_index = min((i + 1) * node_updates_per_pe, total_num_nodes)
        pe_values = values_pool[start_index:end_index]
        pe_values = pe_values.astype(np.int64)
        PEs[i] = pe_values

    if shuffle_fixed == True:
        # Shuffle the keys
        keys = list(PEs.keys())
        values = list(PEs.values())
        shuffled_indices = np.arange(len(keys))
        np.random.shuffle(shuffled_indices)

        # Assign values to shuffled keys
        for i, index in enumerate(shuffled_indices):
            PEs[keys[i]] = values[index]

    current_index = (current_index + total_num_nodes) % len(node_update_schedule)  # int
    return PEs, current_index

# Explicitly specify the return type of the function
# assign_values_to_pes_fixed.signature = types.Tuple((types.DictType(types.int64, types.List(types.int32))), types.int64)

@njit(fastmath=True)
def check_value_in_previous_pes(PEs, pe_id, value, inc_current=False):
    """
    Check if a given value is present in the NumPy array for a specified processing element (PE).
    If inc_current is True, it also checks in the current PE.

    Args:
    PEs (dict): A dictionary where each key is a PE id and each value is a NumPy array of node indices.
    pe_id (int): The processing element id to check the value against.
    value (int): The value (node index) to check for.
    inc_current (bool): Whether to include the current PE's array in the check.

    Returns:
    bool: True if the value is found in the specified PE's array, False otherwise.
    """

    if pe_id in PEs:
        array = PEs[pe_id]  # Directly access the array if the key exists
        return value in array
    else:
        return False  # Return False if the key does not exist

@njit(fastmath=True)
def run_GaBP(A, b, max_iter=100, convergence_threshold=1e-5, show=True):
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    iter_dist_TEST = []

    # run the GaBP iterations
    for iteration in range(max_iter):
        if show:
            print(iteration)
        # get previous state
        prev_P_ij, prev_mu_ij = np.copy(P_ij), np.copy(mu_ij)

        # update messages over all edges
        for edge in edges:
            i, j = edge[0], edge[1]
            P_ij[i][j], mu_ij[i][j] = calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i)

        # get updated state
        curr_P_ij, curr_mu_ij = P_ij, mu_ij

        # get the change of Pᵢⱼ and μᵢⱼ between last and current iteration
        P_ij_change = l2_dist(prev_P_ij, curr_P_ij)
        mu_ij_change = l2_dist(prev_mu_ij, curr_mu_ij)
        total_change = (P_ij_change + mu_ij_change) / num_nodes
        iter_dist[iteration] = total_change

        if show:
            print(total_change)

        iter_dist_TEST.append(total_change)

        # check if average change is good enough to early-stop the algorithm
        if total_change < convergence_threshold:
            print('=> Converged after iteration', iteration+1)
            break

    np.asarray(iter_dist_TEST)
    print("convergence array: ", iter_dist_TEST)

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist_TEST

@njit(fastmath=True)
def run_GaBP_SYNC_ACCELERATED(A, b, max_iter=10, mae=False, convergence_threshold=1e-5, show=False):
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    
    # Initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)
    edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # Track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full(num_nodes, np.nan)

    # List to collect means and standard deviations
    means = np.zeros(max_iter)  # Preallocate space for means
    std_divs = np.zeros(max_iter)  # Preallocate space for standard deviations

    iteration = 0
    early_end = False

    while not early_end and iteration < max_iter:
        # if show:
        #     print(means[:iteration], std_divs[:iteration])

        # Perform updates
        for edge in edges:
            i, j = edge
            P_ij[i, j], mu_ij[i, j] = calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i)

        # Calculate marginals
        P_i = np.zeros(num_nodes)
        mu_i = np.zeros(num_nodes)
        for i in range(num_nodes):
            P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

        # Calculate mean and standard deviation
        current_mean = np.mean(mu_i)
        current_std = np.std(mu_i)
        means[iteration] = current_mean
        std_divs[iteration] = current_std

        # print average
        average = math.sqrt(abs(means[iteration] - means[iteration - 1]))

        if show:
            print(f"iteration: {iteration+1}")
            print(average)     
            print("-----")   

        # Check for convergence
        if iteration > 1 and average < convergence_threshold:
            early_end = True

        iteration += 1

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, std_divs[:iteration], means[:iteration], iteration

def run_GaBP_SYNC(A, b, max_iter=10, mae=False, convergence_threshold=1e-5, show=False):
    
    start = time.time() # -------------------------------------------

    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    mean = [] #np.array([])
    stand_div = [] #np.array([] )
    if show:
        print(mean, stand_div)

    TIME_TO_CHECK = 0

    early_end = False

    prev_s = []
    prev_m = []

    # run the GaBP iterations
    iteration = 0
    while (early_end == False and iteration < max_iter):

        start = time.time()

        if iteration > 2 and max((mean[-1][i] - mean[-2][i]) for i in range(num_nodes)) < convergence_threshold:    # *np.mean(mean):
            break

        # print(iteration)
        # get previous state
        prev_P_ij, prev_mu_ij = np.copy(P_ij), np.copy(mu_ij)

        # update messages over all edges
        for edge in edges:
            i, j = edge[0], edge[1]
            P_ij[i][j], mu_ij[i][j] = calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i)

        # get updated state
        curr_P_ij, curr_mu_ij = P_ij, mu_ij
        
        P = [] # np.array([])
        M = [] # np.array([])
        
        s, m = list(zip(*[calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i) for i in range(num_nodes)]))

        stand_div.append([p for p in s])
        mean.append(m)

        if iteration == 0:
            pass
        else:
            my_result = tuple(map(lambda i, j: math.sqrt(abs(i - j)), m, prev_m))
            end_time = time.time()
            if mae:
                average = sum(my_result) / len(my_result)
                early_end = average < convergence_threshold
                print(f"iteration: {iteration} took {end_time-start} seconds") if mae == False else print(f"iteration: {iteration} took {end_time-start} seconds w/ mae = {'N/A' if average is None else average}")
            else:
                early_end = all(i < convergence_threshold for i in my_result)
                print(f"iteration: {iteration} took {end_time-start} seconds") if mae == False else print(f"iteration: {iteration} took {end_time-start} seconds w/ mae = {'N/A' if average is None else average}")

        prev_s = s
        prev_m = m

        iteration += 1

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    end = time.time()
    TIME_TO_CHECK += (end - start)

    # print("---------------------------------")
    # print("number of iterations: ", iteration)
    # print("---------------------------------")
    # print("time required for convergence: ", TIME_TO_CHECK)
    # print("---------------------------------")

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_div, mean, iteration

def run_GaBP_HARDWARE_BESTCASE(A, b, node_updates_per_stream=1, max_iter=10, TRUE_MEAN = None, mae=False, convergence_threshold=1e-5, show=False):
    
    start = time.time() # -------------------------------------------

    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    TIME_TO_CHECK = 0

    early_end = False

    list_of_all_nodes = [i for i in range(0, len(b))]

    # run the GaBP iterations
    iteration = 0

    # list of standard deviations and means
    STD, MEAN = np.array(list(zip(*[calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i) for i in range(num_nodes)])))

    while (early_end == False and iteration < max_iter):
        
        start_time = time.time()

        # update messages over all edges
        nodes_to_update = random.sample(list_of_all_nodes, min(node_updates_per_stream, len(b)))

        temp_P_ij = np.copy(P_ij)
        temp_mu_ij = np.copy(mu_ij)

        for node in nodes_to_update:
            edges = find_edges_UPDATED(A, node)
            for edge in edges:
                x, y = edge[1], edge[0]
                # if x == node:
                #     x,y = y,x
                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
            
            # STD[node], MEAN[node] = calc_node_marginal(node, P_ii, mu_ii, P_ij, mu_ij, N_i)

        STD, MEAN = np.array(list(zip(*[calc_node_marginal(i, P_ii, mu_ii, temp_P_ij, temp_mu_ij, N_i) for i in range(num_nodes)])))

        P_ij, mu_ij = np.copy(temp_P_ij), np.copy(temp_mu_ij)

        end_time = time.time()

        if TRUE_MEAN == None:
            if iteration > max_iter:
                break
        else:
            result = (tuple(map(lambda i, j: math.sqrt(abs(i - j)), MEAN, TRUE_MEAN)))
            if mae:
                average = sum(result) / len(result)
                early_end = average < convergence_threshold
                print(f"iteration: {iteration} took {end_time-start_time} seconds") if mae == False else print(f"iteration: {iteration} took {end_time-start_time} seconds w/ mae = {'N/A' if average is None else average}")
            else:
                early_end = all(i < convergence_threshold for i in result)
                print(f"iteration: {iteration} took {end_time-start_time} seconds") if mae == False else print(f"iteration: {iteration} took {end_time-start_time} seconds w/ mae = {'N/A' if average is None else average}")

        iteration += 1

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    end = time.time()
    TIME_TO_CHECK += (end - start)

    # print("---------------------------------")
    # print("number of iterations: ", iteration)
    # print("---------------------------------")
    # print("time required for convergence: ", TIME_TO_CHECK)
    # print("---------------------------------")

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, [STD], [MEAN], iteration+1

def run_GaBP_HARDWARE_WORSTCASE(A, b, TRUE_MEAN, node_updates_per_stream = 3, max_iter=10, mae=True, convergence_threshold=1e-5, mode = 'random', residual=None, node_update_schedule=None, show=False):
    
    nodes_to_be_updt = node_updates_per_stream
    mean_arr = TRUE_MEAN
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    # edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    mean = [] #np.array([])
    stand_div = [] #np.array([] )
    if show:
        print(mean, stand_div)

    # run the GaBP iterations
    iteration = 0
    early_end = False

    TIME_TO_CHECK = 0

    if mode == 'fixed' or mode == 'residual':
        schedule_index = 0

    # residual checker
    residual_container = {i: float("inf") for i in range(len(mean_arr))}

    # temperoary dict
    my_queue = queue.Queue()
    for i in range(len(mean_arr)):
        my_queue.put(i)

    # nodes updates
    updated_nodes = {i: 0 for i in range(len(mean_arr))}

    # node updated schedule
    node_update_schedule = random.sample(range(num_nodes), num_nodes) if node_update_schedule == None else node_update_schedule

    # pass identifier
    pass_dict = {}
    pass_counter = 1

    out_str = ""

    """Updated code"""
    INDEX = -1
    # copy 1
    P_ij_copy_1 = np.copy(P_ij)
    mu_ij_copy_1 = np.copy(mu_ij)

    # copy 2
    P_ij_copy_2 = np.copy(P_ij)
    mu_ij_copy_2 = np.copy(mu_ij)

    while (early_end == False):

        if mode == 'random':

            if iteration == 0:
                
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)

                INDEX = 1

            else:
                if INDEX == 1:
                    
                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_1)
                    temp_mu_ij = np.copy(mu_ij_copy_1)

                    random_integers = random.sample(range(len(mean_arr)), nodes_to_be_updt)

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)

                        updated_nodes[num] += 1

                        # append to pass dict
                        pass_dict[pass_counter].append(num)
                        
                        for edge in edges:
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)

                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_2 = np.copy(P_ij)
                    mu_ij_copy_2 = np.copy(mu_ij)

                    # ---------------- write back to stream ----------------

                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for edge in edges:
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 2
                
                elif INDEX == 2:

                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_2)
                    temp_mu_ij = np.copy(mu_ij_copy_2)

                    random_integers = random.sample(range(len(mean_arr)), nodes_to_be_updt)

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)

                        updated_nodes[num] += 1

                        # append to pass dict
                        pass_dict[pass_counter].append(num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)

                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_1 = np.copy(P_ij)
                    mu_ij_copy_1 = np.copy(mu_ij)
                    
                    # ---------------- write back to stream  ----------------
                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 1

        elif mode == 'fixed':
            
            if iteration == 0:
                
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)

                INDEX = 1

            else:
                if INDEX == 1:

                    
                    schedule_index_tmp = schedule_index
                    
                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_1)
                    temp_mu_ij = np.copy(mu_ij_copy_1)

                    random_integers = random.sample(range(len(mean_arr)), nodes_to_be_updt)

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    for iter in range(nodes_to_be_updt):

                        num = node_update_schedule[schedule_index_tmp]
                        schedule_index_tmp = (schedule_index_tmp + 1) % len(node_update_schedule)
                        edges = find_edges_UPDATED(A, num)

                        updated_nodes[num] += 1

                        # append to pass dict
                        pass_dict[pass_counter].append(num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)

                        # print(num)

                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_2 = np.copy(P_ij)
                    mu_ij_copy_2 = np.copy(mu_ij)

                    # ---------------- write back to stream ----------------

                    for iter in range(nodes_to_be_updt):

                        num = node_update_schedule[schedule_index]
                        schedule_index = (schedule_index + 1) % len(node_update_schedule)
                        edges = find_edges_UPDATED(A, num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]
                        
                        # print(num)

                    # ---------------- updated index ----------------
                    INDEX = 2
                
                elif INDEX == 2:

                    schedule_index_tmp = schedule_index

                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_2)
                    temp_mu_ij = np.copy(mu_ij_copy_2)

                    random_integers = random.sample(range(len(mean_arr)), nodes_to_be_updt)

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    for iter in range(nodes_to_be_updt):

                        num = node_update_schedule[schedule_index_tmp]
                        schedule_index_tmp = (schedule_index_tmp + 1) % len(node_update_schedule)
                        edges = find_edges_UPDATED(A, num)

                        updated_nodes[num] += 1

                        # append to pass dict
                        pass_dict[pass_counter].append(num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)

                        # print(num)

                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_1 = np.copy(P_ij)
                    mu_ij_copy_1 = np.copy(mu_ij)

                    
                    # ---------------- write back to stream  ----------------
                    for iter in range(nodes_to_be_updt):

                        num = node_update_schedule[schedule_index]
                        schedule_index = (schedule_index + 1) % len(node_update_schedule)
                        edges = find_edges_UPDATED(A, num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[0], edge[1]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                        # print(num)

                    # ---------------- updated index ----------------
                    INDEX = 1
        

        
        s, m = list(zip(*[calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i) for i in range(num_nodes)]))

        stand_div.append([p for p in s])
        mean.append(m)

        my_result = tuple(map(lambda i, j: math.sqrt(abs(i - j)), m, mean_arr))

        out_str += f"my result at iteration {iteration} = {str(my_result)}\n-----------------\n"

        start = time.time()
        if mae:
            average = sum(my_result) / len(my_result)
            early_end = average < convergence_threshold
        else:
            early_end = all(i < convergence_threshold for i in my_result)
        end = time.time()
        TIME_TO_CHECK += (end - start)

        iteration += 1

        # if show:
        print(f"iteration: {iteration} took {TIME_TO_CHECK} seconds")
            # print(out_str)

    if show:
        print("mean:", np.array(mean))
        print("standard deviation:", np.array(stand_div))

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_div, mean, iteration+1, updated_nodes, pass_dict

def run_GaBP_HARDWARE(A, b, TRUE_MEAN, node_updates_per_pe = 1, number_pes = 1, max_iter=10, mae=True, convergence_threshold=1e-5, mode = 'random', residual=None, node_update_schedule=None, show=False):
    
    nodes_to_be_updt = node_updates_per_pe*number_pes
    mean_arr = TRUE_MEAN
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    # edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    mean = [] #np.array([])
    stand_div = [] #np.array([] )
    if show:
        print(mean, stand_div)

    # run the GaBP iterations
    iteration = 0
    early_end = False

    TIME_TO_CHECK = 0

    if mode == 'fixed' or mode == 'residual':
        schedule_index = 0

    # residual checker
    residual_container = {i: float("inf") for i in range(len(mean_arr))}

    # temperoary dict
    my_queue = queue.Queue()
    for i in range(len(mean_arr)):
        my_queue.put(i)

    # nodes updates
    updated_nodes = {i: 0 for i in range(len(mean_arr))}

    # node updated schedule
    node_update_schedule = random.sample(range(num_nodes), num_nodes) if node_update_schedule == None else node_update_schedule

    # pass identifier
    pass_dict = {}
    pass_counter = 1

    out_str = ""

    """Updated code"""
    INDEX = -1
    # copy 1
    P_ij_copy_1 = np.copy(P_ij)
    mu_ij_copy_1 = np.copy(mu_ij)

    # copy 2
    P_ij_copy_2 = np.copy(P_ij)
    mu_ij_copy_2 = np.copy(mu_ij)

    # dictionary of all pes
    PEs_part_1 = {key: [] for key in range(0, num_nodes)}
    PEs_part_2 = {key: [] for key in range(0, num_nodes)}

    while (early_end == False):

        start_time = time.time()

        if mode == 'random':

            if iteration == 0:
                
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)

                PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                INDEX = 1

            else:
                if INDEX == 1:
                    
                    PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_1)
                    temp_mu_ij = np.copy(mu_ij_copy_1)

                    random_integers = [value for sublist in PEs_part_1.values() for value in sublist]

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    # ---------------- update for PE 0 ----------------

                    # iterate through nodes, and check if a node is in memory
                    # nodes = PEs_part_1[0]

                    # find nodes in pe or before
                    nodes_in_pe_or_before = set()
                    for pe_id, nodes in PEs_part_1.items():
                        for node in nodes:
                            edges = find_edges_UPDATED(A, node)
                            local_node_edges = []
                            for edge in edges:
                                x, y = edge[1], edge[0]
                                if x == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=y, inc_current=True):
                                        nodes_in_pe_or_before.add(tuple(edge))
                                        local_node_edges.append(tuple(edge))
                                if y == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=x, inc_current=True):
                                        nodes_in_pe_or_before.add(tuple(edge))
                                        local_node_edges.append(tuple(edge))
                            print(local_node_edges)
                            print("------------------------")

                    # get all neighbors edge
                    edges = find_edges(A)
                    for edge in edges:
                            x, y = edge[1], edge[0]
                            if tuple(edge) in nodes_in_pe_or_before:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                            else:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)
                    
                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_2 = np.copy(P_ij)
                    mu_ij_copy_2 = np.copy(mu_ij)

                    # ---------------- write back to stream ----------------

                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for edge in edges:
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 2
                
                elif INDEX == 2:

                    # ---------------- update nodes using INDEX 1 ----------------

                    PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_2)
                    temp_mu_ij = np.copy(mu_ij_copy_2)

                    random_integers = [value for sublist in PEs_part_2.values() for value in sublist]

                    # create pass dict entry
                    pass_dict[pass_counter] = []

                    # ---------------- update for PE 0 ----------------

                    # find nodes in pe or before
                    nodes_in_pe_or_before = set()
                    for pe_id, nodes in PEs_part_2.items():
                        for node in nodes:
                            edges = find_edges_UPDATED(A, node)
                            local_node_edges = []
                            for edge in edges:
                                x, y = edge[1], edge[0]
                                if x == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=y, inc_current=True):
                                        nodes_in_pe_or_before.add(tuple(edge))
                                        local_node_edges.append(tuple(edge))
                                if y == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=x, inc_current=True):
                                        nodes_in_pe_or_before.add(tuple(edge))
                                        local_node_edges.append(tuple(edge))
                            print(local_node_edges)
                            print("------------------------")

                    # get all neighbors edge
                    edges = find_edges(A)
                    for edge in edges:
                            x, y = edge[1], edge[0]
                            if tuple(edge) in nodes_in_pe_or_before:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                            else:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)
                    
                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_1 = np.copy(P_ij)
                    mu_ij_copy_1 = np.copy(mu_ij)
                    
                    # ---------------- write back to stream  ----------------
                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 1
   
        s, m = list(zip(*[calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i) for i in range(num_nodes)]))

        stand_div.append([p for p in s])
        mean.append(m)

        my_result = tuple(map(lambda i, j: math.sqrt(abs(i - j)), m, mean_arr))
    
        end_time = time.time()
        TIME_TO_CHECK = (end_time - start_time)

        if mae:
            average = sum(my_result) / len(my_result)
            early_end = average < convergence_threshold
            if show:
                print(f"iteration: {iteration} took {TIME_TO_CHECK} seconds: mae = {average}")            
        else:
            average = sum(my_result) / len(my_result)
            early_end = all(i < convergence_threshold for i in my_result)
            if show:
                print(f"iteration: {iteration} took {TIME_TO_CHECK} seconds: mae = {average}")

        iteration += 1

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_div, mean, iteration+1, updated_nodes, pass_dict

@njit(fastmath=True)
def custom_unique(rows):
    if rows.size == 0:
        return rows  # Return empty array directly if input is empty

    # Initialize unique_rows with the first row of 'rows' to ensure dimension compatibility
    unique_rows = rows[0:1]
    for i in range(1, len(rows)):
        row = rows[i:i+1]  # Slice to keep row as a 2D array
        match_found = False
        for unique in unique_rows:
            if np.array_equal(row, unique):
                match_found = True
                break
        if not match_found:
            unique_rows = np.vstack((unique_rows, row))
    
    return unique_rows

@njit(fastmath=True)
def random_sample_set(arr, k=1):
    # print(" ---- START: entering random_sample_set ---- ")
    # print(f"arr = {arr}")
    # print(type(arr))
    # print(f"arr.shape[0] = {arr.shape[0]}")
    # print(f"arr.size = {arr.size}")
    index = np.random.choice(np.arange(arr.shape[0]), size=k, replace=False)
    # print(" ---- END: exiting random_sample_set ---- ")
    return arr[index]

@njit(fastmath=True)
def run_GaBP_HARDWARE_ACCELERATED_OLD(A, b, TRUE_MEAN, node_updates_per_pe = 1, number_pes = 1, max_iter=10, mae=True, convergence_threshold=1e-5, mode = 'random', residual=None, node_update_schedule=None, show=False):
    
    mean_arr = np.array(TRUE_MEAN)
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # run the GaBP iterations
    iteration = 0
    early_end = False

    """Updated code"""
    INDEX = -1
    # copy 1
    P_ij_copy_1 = np.copy(P_ij)
    mu_ij_copy_1 = np.copy(mu_ij)

    # copy 2
    P_ij_copy_2 = np.copy(P_ij)
    mu_ij_copy_2 = np.copy(mu_ij)

    # dictionary of all pes
    PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 
    PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

    while (early_end == False):

        if mode == 'random':

            if iteration == 0:
                
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)

                PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                INDEX = 1

            else:
                if INDEX == 1:
                    
                    PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                    # ---------------- update nodes using INDEX 1 ----------------

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_1)
                    temp_mu_ij = np.copy(mu_ij_copy_1)

                    random_integers = [value for sublist in PEs_part_1.values() for value in sublist]

                    # ---------------- update for PE 0 ----------------

                    # find nodes in pe or before
                    nodes_in_pe_or_before = []
                    for pe_id, nodes in PEs_part_1.items():
                        for node in nodes:
                            edges = find_edges_UPDATED(A, node)
                            for edge in edges:
                                x, y = edge[1], edge[0]
                                if x == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=y, inc_current=False):
                                        nodes_in_pe_or_before.append(edge)
                                if y == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=x, inc_current=False):
                                        nodes_in_pe_or_before.append(edge)
                    
                    if nodes_in_pe_or_before:  # Checks if the list is not empty
                        nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                        nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                    # get all neighbors edge
                    edges = find_edges(A)
                    for edge in edges:
                            x, y = edge[1], edge[0]
                            target_tuple = edge

                            match_found = False

                            if nodes_in_pe_or_before:
                                for i in range(nodes_in_pe_or_before_np.shape[0]):
                                    if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                        match_found = True
                                        break

                            if match_found:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                            else:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)
                    
                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_2 = np.copy(P_ij)
                    mu_ij_copy_2 = np.copy(mu_ij)

                    # ---------------- write back to stream ----------------

                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for edge in edges:
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 2
                
                elif INDEX == 2:

                    # ---------------- update nodes using INDEX 1 ----------------

                    PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                             node_updates_per_pe=node_updates_per_pe, 
                                             number_pes=number_pes) 

                    # Create temporary arrays to store updated messages
                    temp_P_ij = np.copy(P_ij_copy_2)
                    temp_mu_ij = np.copy(mu_ij_copy_2)

                    random_integers = [value for sublist in PEs_part_2.values() for value in sublist]

                    # ---------------- update for PE 0 ----------------

                    # find nodes in pe or before
                    nodes_in_pe_or_before = []
                    for pe_id, nodes in PEs_part_2.items():
                        for node in nodes:
                            edges = find_edges_UPDATED(A, node)
                            for edge in edges:
                                x, y = edge[1], edge[0]
                                if x == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=y, inc_current=False):
                                        nodes_in_pe_or_before.append(edge)
                                if y == node:
                                    if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=x, inc_current=False):
                                        nodes_in_pe_or_before.append(edge)

                    if nodes_in_pe_or_before:  # Checks if the list is not empty
                        nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                        nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                    # get all neighbors edge
                    edges = find_edges(A)
                    for edge in edges:
                            x, y = edge[1], edge[0]
                            target_tuple = edge

                            match_found = False

                            if nodes_in_pe_or_before:
                                for i in range(nodes_in_pe_or_before_np.shape[0]):
                                    if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                        match_found = True
                                        break

                            if match_found:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                            else:
                                temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)
                    
                    # ---------------- collect for next iteration ----------------
                    P_ij_copy_1 = np.copy(P_ij)
                    mu_ij_copy_1 = np.copy(mu_ij)
                    
                    # ---------------- write back to stream  ----------------
                    for iter in random_integers:

                        num = iter
                        edges = find_edges_UPDATED(A, num)
                        
                        for i in range(np.shape(edges)[0]):
                            edge = edges[i]
                            x, y = edge[1], edge[0]

                            # Update temporary messages
                            P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                    # ---------------- updated index ----------------
                    INDEX = 1

        # Calculate marginals
        s = np.zeros(num_nodes)
        m = np.zeros(num_nodes)
        for i in range(num_nodes):
            s[i], m[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

        my_result = np.sqrt(np.abs(m - mean_arr))
        
        
        if mae:
            average = np.mean(my_result)
            early_end = average < convergence_threshold
            if show:
                print(f"iteration: {iteration+1}")
                print(average)     
                print("-----")       
        else:
            early_end = np.all(my_result < convergence_threshold) # all(i < convergence_threshold for i in my_result)
            if show:
                print(f"iteration: {iteration+1}")
                print(np.max(my_result))     
                print("-----") 

        iteration += 1

        # # calculate marginals
        P_i = np.zeros_like(P_ii)
        mu_i = np.zeros_like(mu_ii)
        for i in range(num_nodes):
            P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, iteration+1

@njit(fastmath=True)
def run_GaBP_HARDWARE_ACCELERATED(A, b, TRUE_MEAN, capping=None, node_updates_per_pe=1, number_pes=1, shuffled_fixed=True, max_iter=1000, mae=True, convergence_threshold=1e-5, mode='random', node_update_schedule_enter=None, show=False):
    
    mean_arr = np.array(TRUE_MEAN)
    
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # run the GaBP iterations
    iteration = 0
    early_end = False
    
    if node_update_schedule_enter == None:
        node_update_schedule = np.arange(num_nodes)
    else:
        node_update_schedule = node_update_schedule_enter  

    """Updated code"""
    INDEX = -1
    # copy 1
    P_ij_copy_1 = np.copy(P_ij)
    mu_ij_copy_1 = np.copy(mu_ij)

    # copy 2
    P_ij_copy_2 = np.copy(P_ij)
    mu_ij_copy_2 = np.copy(mu_ij)

    current_index = 0

    # dictionary of all pes
    if mode == 'random':
        PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes) 
        PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes)
    elif mode == 'fixed':
        PEs_part_1, current_index = assign_values_to_pes_fixed(num_nodes=num_nodes, node_update_schedule=node_update_schedule, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes,
                                                 current_index=current_index,
                                                 shuffle_fixed=shuffled_fixed) 
        
        PEs_part_2, current_index = assign_values_to_pes_fixed(num_nodes=num_nodes, node_update_schedule=node_update_schedule, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes,
                                                 current_index=current_index,
                                                 shuffle_fixed=shuffled_fixed) 
        
        
    while (early_end == False):

        if iteration == 0:
            
            P_ij_copy_1 = np.copy(P_ij)
            mu_ij_copy_1 = np.copy(mu_ij)

            # dictionary of all pes
            if mode == 'random':
                PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                         node_updates_per_pe=node_updates_per_pe, 
                                                         number_pes=number_pes) 
            elif mode == 'fixed':
                PEs_part_2, current_index = assign_values_to_pes_fixed(num_nodes=num_nodes, node_update_schedule=node_update_schedule, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes,
                                                 current_index=current_index,
                                                 shuffle_fixed=shuffled_fixed)
                
            INDEX = 1

        else:
            if INDEX == 1:
                
                # dictionary of all pes
                if mode == 'random':
                    PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                             node_updates_per_pe=node_updates_per_pe, 
                                                             number_pes=number_pes) 
                elif mode == 'fixed':
                    PEs_part_1, current_index = assign_values_to_pes_fixed(num_nodes=num_nodes, node_update_schedule=node_update_schedule, 
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes,
                                                 current_index=current_index,
                                                 shuffle_fixed=shuffled_fixed)
                    
                # ---------------- update nodes using INDEX 1 ----------------

                # Create temporary arrays to store updated messages
                temp_P_ij = np.copy(P_ij_copy_1)
                temp_mu_ij = np.copy(mu_ij_copy_1)

                random_integers = [value for sublist in PEs_part_1.values() for value in sublist]

                # ---------------- update for PE 0 ----------------

                # find nodes in pe or before
                nodes_in_pe_or_before = []
                for pe_id, nodes in PEs_part_1.items():
                    for node in nodes:
                        edges = find_edges_UPDATED(A, node)
                        for edge in edges:
                            x, y = edge[1], edge[0]
                            if x == node:
                                if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=y, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                            if y == node:
                                if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=x, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                
                if nodes_in_pe_or_before:  # Checks if the list is not empty
                    nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                    nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                # get all neighbors edge
                edges = find_edges(A)
                for edge in edges:
                        x, y = edge[1], edge[0]
                        target_tuple = edge

                        match_found = False

                        if nodes_in_pe_or_before:
                            for i in range(nodes_in_pe_or_before_np.shape[0]):
                                if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                    match_found = True
                                    break

                        if match_found:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                        else:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)
                
                # ---------------- collect for next iteration ----------------
                P_ij_copy_2 = np.copy(P_ij)
                mu_ij_copy_2 = np.copy(mu_ij)

                # ---------------- write back to stream  ----------------
                for iter in random_integers:

                    num = iter
                    edges = find_edges_UPDATED(A, num)
                    
                    edges_np = np.array(edges)
                    edges_np = random_sample_set(edges_np, k=min(capping, np.shape(edges_np)[0])) if capping != None else edges_np

                    for i in range(np.shape(edges_np)[0]):
                        edge = edges_np[i]
                        x, y = edge[1], edge[0]

                        # Update temporary messages
                        P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                # ---------------- updated index ----------------
                INDEX = 2
            
            elif INDEX == 2:

                # ---------------- update nodes using INDEX 1 ----------------

                            # dictionary of all pes
                if mode == 'random':
                    PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                             node_updates_per_pe=node_updates_per_pe, 
                                                             number_pes=number_pes)
                elif mode == 'fixed':
                    PEs_part_2, current_index = assign_values_to_pes_fixed(num_nodes=num_nodes, node_update_schedule=node_update_schedule,
                                                 node_updates_per_pe=node_updates_per_pe, 
                                                 number_pes=number_pes,
                                                 current_index=current_index,
                                                 shuffle_fixed=shuffled_fixed)
                    
                # Create temporary arrays to store updated messages
                temp_P_ij = np.copy(P_ij_copy_2)
                temp_mu_ij = np.copy(mu_ij_copy_2)

                random_integers = [value for sublist in PEs_part_2.values() for value in sublist]

                # ---------------- update for PE 0 ----------------

                # find nodes in pe or before
                nodes_in_pe_or_before = []
                for pe_id, nodes in PEs_part_2.items():
                    for node in nodes:
                        edges = find_edges_UPDATED(A, node)
                        for edge in edges:
                            x, y = edge[1], edge[0]
                            if x == node:
                                if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=y, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                            if y == node:
                                if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=x, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)

                if nodes_in_pe_or_before:  # Checks if the list is not empty
                    nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                    nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                # get all neighbors edge
                edges = find_edges(A)
                for edge in edges:
                        x, y = edge[1], edge[0]
                        target_tuple = edge

                        match_found = False

                        if nodes_in_pe_or_before:
                            for i in range(nodes_in_pe_or_before_np.shape[0]):
                                if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                    match_found = True
                                    break

                        if match_found:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                        else:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)
                
                # ---------------- collect for next iteration ----------------
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)
                
                # ---------------- write back to stream  ----------------
                for iter in random_integers:

                    num = iter
                    edges = find_edges_UPDATED(A, num)
                    
                    edges_np = np.array(edges)
                    edges_np = random_sample_set(edges_np, k=min(capping, np.shape(edges_np)[0])) if capping != None else edges_np

                    for i in range(np.shape(edges_np)[0]):
                        edge = edges_np[i]
                        x, y = edge[1], edge[0]

                        # Update temporary messages
                        P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                # ---------------- updated index ----------------
                INDEX = 1

        # Calculate marginals
        s = np.zeros(num_nodes)
        m = np.zeros(num_nodes)
        for i in range(num_nodes):
            s[i], m[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

        my_result = np.sqrt(np.abs(m - mean_arr))
        
        
        if mae:
            average = np.mean(my_result)
            early_end = average < convergence_threshold
            if show:
                print(f"iteration: {iteration+1}")
                print(average)     
                print("-----")       
        else:
            early_end = np.all(my_result < convergence_threshold) # all(i < convergence_threshold for i in my_result)
            if show:
                print(f"iteration: {iteration+1}")
                print(np.max(my_result))     
                print("-----") 

        iteration += 1

        # # calculate marginals
        P_i = np.zeros_like(P_ii)
        mu_i = np.zeros_like(mu_ii)
        for i in range(num_nodes):
            P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, iteration+1

@njit(fastmath=True)
def run_GaBP_HARDWARE_ACCELERATED_RESIDUAL_COARSE(A, b, TRUE_MEAN, capping=None, node_updates_per_pe=1, number_pes=1, shuffled_fixed=True, max_iter=1000, mae=True, convergence_threshold=1e-5, mode='random', node_update_schedule_enter=None, show=False):
    
    mean_arr = np.array(TRUE_MEAN)
    
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # run the GaBP iterations
    iteration = 0
    early_end = False
    
    # (delete me) node update schdule
    if node_update_schedule_enter == None:
        node_update_schedule = np.arange(num_nodes)
    else:
        node_update_schedule = node_update_schedule_enter 

    residual_container = {}
    for i in range(len(mean_arr)):
        residual_container[i] = np.inf

    print(residual_container)

    """Updated code"""
    INDEX = -1
    # copy 1
    P_ij_copy_1 = np.copy(P_ij)
    mu_ij_copy_1 = np.copy(mu_ij)

    # copy 2
    P_ij_copy_2 = np.copy(P_ij)
    mu_ij_copy_2 = np.copy(mu_ij)

    while (early_end == False):

        if iteration == 0:
            
            P_ij_copy_1 = np.copy(P_ij)
            mu_ij_copy_1 = np.copy(mu_ij)

            PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                        node_updates_per_pe=node_updates_per_pe, 
                                                        number_pes=number_pes) 
                
            INDEX = 1

        else:
            if INDEX == 1:
                
                PEs_part_1 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                            node_updates_per_pe=node_updates_per_pe, 
                                                            number_pes=number_pes) 
                    
                # ---------------- update nodes using INDEX 1 ----------------

                # Create temporary arrays to store updated messages
                temp_P_ij = np.copy(P_ij_copy_1)
                temp_mu_ij = np.copy(mu_ij_copy_1)

                random_integers = [value for sublist in PEs_part_1.values() for value in sublist]

                # ---------------- update for PE 0 ----------------

                # find nodes in pe or before
                nodes_in_pe_or_before = []
                for pe_id, nodes in PEs_part_1.items():
                    for node in nodes:
                        edges = find_edges_UPDATED(A, node)
                        for edge in edges:
                            x, y = edge[1], edge[0]
                            if x == node:
                                if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=y, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                            if y == node:
                                if check_value_in_previous_pes(PEs=PEs_part_2, pe_id=pe_id, value=x, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                
                if nodes_in_pe_or_before:  # Checks if the list is not empty
                    nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                    nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                # get all neighbors edge
                edges = find_edges(A)
                for edge in edges:
                        x, y = edge[1], edge[0]
                        target_tuple = edge

                        match_found = False

                        if nodes_in_pe_or_before:
                            for i in range(nodes_in_pe_or_before_np.shape[0]):
                                if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                    match_found = True
                                    break

                        if match_found:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                        else:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_1, mu_ij_copy_1, N_i)
                
                # ---------------- collect for next iteration ----------------
                P_ij_copy_2 = np.copy(P_ij)
                mu_ij_copy_2 = np.copy(mu_ij)

                # ---------------- write back to stream  ----------------
                for iter in random_integers:

                    num = iter
                    edges = find_edges_UPDATED(A, num)
                    
                    edges_np = np.array(edges)
                    edges_np = random_sample_set(edges_np, k=min(capping, np.shape(edges_np)[0])) if capping != None else edges_np

                    for i in range(np.shape(edges_np)[0]):
                        edge = edges_np[i]
                        x, y = edge[1], edge[0]

                        # Update temporary messages
                        P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                # ---------------- updated index ----------------
                INDEX = 2
            
            elif INDEX == 2:

                # ---------------- update nodes using INDEX 1 ----------------

                # dictionary of all pes
                PEs_part_2 = assign_values_to_pes_random(num_nodes=num_nodes, 
                                                            node_updates_per_pe=node_updates_per_pe, 
                                                            number_pes=number_pes)
                    
                # Create temporary arrays to store updated messages
                temp_P_ij = np.copy(P_ij_copy_2)
                temp_mu_ij = np.copy(mu_ij_copy_2)

                random_integers = [value for sublist in PEs_part_2.values() for value in sublist]

                # ---------------- update for PE 0 ----------------

                # find nodes in pe or before
                nodes_in_pe_or_before = []
                for pe_id, nodes in PEs_part_2.items():
                    for node in nodes:
                        edges = find_edges_UPDATED(A, node)
                        for edge in edges:
                            x, y = edge[1], edge[0]
                            if x == node:
                                if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=y, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)
                            if y == node:
                                if check_value_in_previous_pes(PEs=PEs_part_1, pe_id=pe_id, value=x, inc_current=False):
                                    nodes_in_pe_or_before.append(edge)

                if nodes_in_pe_or_before:  # Checks if the list is not empty
                    nodes_in_pe_or_before_np = np.array(nodes_in_pe_or_before)
                    nodes_in_pe_or_before_np = custom_unique(nodes_in_pe_or_before_np)

                # get all neighbors edge
                edges = find_edges(A)
                for edge in edges:
                        x, y = edge[1], edge[0]
                        target_tuple = edge

                        match_found = False

                        if nodes_in_pe_or_before:
                            for i in range(nodes_in_pe_or_before_np.shape[0]):
                                if np.all(nodes_in_pe_or_before_np[i] == target_tuple):
                                    match_found = True
                                    break

                        if match_found:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij, mu_ij, N_i)
                        else:
                            temp_P_ij[x][y], temp_mu_ij[x][y] = calc_m_ij(x, y, A, P_ii, mu_ii, P_ij_copy_2, mu_ij_copy_2, N_i)
                
                # ---------------- collect for next iteration ----------------
                P_ij_copy_1 = np.copy(P_ij)
                mu_ij_copy_1 = np.copy(mu_ij)
                
                # ---------------- write back to stream  ----------------
                for iter in random_integers:

                    num = iter
                    edges = find_edges_UPDATED(A, num)
                    
                    edges_np = np.array(edges)
                    edges_np = random_sample_set(edges_np, k=min(capping, np.shape(edges_np)[0])) if capping != None else edges_np

                    for i in range(np.shape(edges_np)[0]):
                        edge = edges_np[i]
                        x, y = edge[1], edge[0]

                        # Update temporary messages
                        P_ij[x][y], mu_ij[x][y] = temp_P_ij[x][y], temp_mu_ij[x][y]

                # ---------------- updated index ----------------
                INDEX = 1

        # Calculate marginals
        s = np.zeros(num_nodes)
        m = np.zeros(num_nodes)
        for i in range(num_nodes):
            s[i], m[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

        my_result = np.sqrt(np.abs(m - mean_arr))
        
        
        if mae:
            average = np.mean(my_result)
            early_end = average < convergence_threshold
            if show:
                print(f"iteration: {iteration+1}")
                print(average)     
                print("-----")       
        else:
            early_end = np.all(my_result < convergence_threshold) # all(i < convergence_threshold for i in my_result)
            if show:
                print(f"iteration: {iteration+1}")
                print(np.max(my_result))     
                print("-----") 

        iteration += 1

        # # calculate marginals
        P_i = np.zeros_like(P_ii)
        mu_i = np.zeros_like(mu_ii)
        for i in range(num_nodes):
            P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, iteration+1