import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime
import argparse
import threading
import multiprocessing
import os

from tqdm import tqdm

from GBP_Simulations.GBP.data import DataGenerator
from GBP_Simulations.GBP.gbp import run_GaBP_SYNC_ACCELERATED, run_GaBP_HARDWARE_ACCELERATED, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL
from GBP_Simulations.GBP.utilities import HiddenPrints
from GBP_Simulations.GBP.visulisation import set_plot_options, get_plot_colors, NetworkxGraph, AnalyzeResult


def process_designs(designs, num_iterations, progress_queue=None, output_list=None):

    output_designs = []

    for design in designs:
        node_updates_per_pe = design['nodes_updt_per_pe']
        number_pes = design['number_pes']
        policy = design['policy']
        cache = design['cache']
        sum_of_iterations = 0
        for it in range(num_iterations):
            if policy == 'fixed':
                num_nodes = A.shape[0]
                node_update_schedule = np.arange(num_nodes, dtype=np.int64)
                P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED(
                    A, 
                    b, 
                    node_updates_per_pe=node_updates_per_pe, 
                    number_pes=number_pes, 
                    TRUE_MEAN=final_mean, 
                    max_iter=10000, 
                    mae=True, 
                    convergence_threshold=simulation_convergence_threshold,
                    show=False,
                    mode=policy,
                    caching=cache,
                    node_update_schedule_enter=node_update_schedule,
                    shuffled_fixed=True
                )
                if iteration < 10000:
                    break
                else:
                    iteration = float("inf")
                    print("=========== FIXED: NOT CONVERGING ===========")
            elif policy == 'random':
                P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED(
                    A, 
                    b, 
                    node_updates_per_pe=node_updates_per_pe, 
                    number_pes=number_pes, 
                    TRUE_MEAN=final_mean, 
                    max_iter=10000, 
                    mae=True, 
                    convergence_threshold=simulation_convergence_threshold,
                    show=False,
                    mode=policy,
                    caching=cache
                )
                if iteration < 10000:
                    break
                else:
                    iteration = float("inf")
                    print("=========== RANDOM-EXCLUSION: NOT CONVERGING ===========")
            elif policy == 'random-exclusion':
                P_i, mu_i, iteration, _ = run_GaBP_HARDWARE_ACCELERATED_EXCLUSION(
                    A, 
                    b, 
                    caching=cache,
                    node_updates_per_pe=node_updates_per_pe, 
                    number_pes=number_pes, 
                    TRUE_MEAN=final_mean, 
                    max_iter=10000, 
                    mae=False, 
                    convergence_threshold=simulation_convergence_threshold, 
                    show=False
                )
                if iteration < 10000:
                    break
                else:
                    iteration = float("inf")
                    print("=========== RANDOM: NOT CONVERGING ===========")
            elif policy == 'residual':
                P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED_RESIDUAL(
                    A, 
                    b, 
                    caching=cache,
                    node_updates_per_pe=node_updates_per_pe, 
                    number_pes=number_pes, 
                    TRUE_MEAN=final_mean, 
                    max_iter=10000, 
                    mae=False, 
                    convergence_threshold=simulation_convergence_threshold, 
                    show=False
                )
                if iteration < 10000:
                    break
                else:
                    iteration = float("inf")
                    print("=========== RESIDUAL: NOT CONVERGING ===========")
            else:
                print(policy)
                print("----------------")
                raise Exception("Error: No Matching Policy")
            sum_of_iterations += iteration
            if progress_queue == None:
                pass
            else:
                progress_queue.put(1)
        ave_convergence = sum_of_iterations / num_iterations
        design['stream_passes'] = ave_convergence
        output_designs.append(design)

    if output_list is not None:
        output_list.extend(output_designs)
    else:
        return output_designs

def update_progress(progress_queue, total):
    while True:
        if progress_queue.empty():
            if progress_queue._closed:
                break
            continue
        progress_queue.get()
        pbar.update(1)
    pbar.close()

# Load the JSON data

parser = argparse.ArgumentParser(description='Generate Designs')
parser.add_argument('-save', type=str, help='name of file to be saved', default="")
filename = parser.parse_args().save

filename = "slam_1d_1000"
with open(f"Hardware_Model/designs/{filename}.json") as f:
    data = json.load(f)

# Extract designs
graph = data.get('inp_graph_topology', [])
designs = data.get('all_designs', [])

# Create a set to store unique combinations
unique_combinations = set()

# Initialize lists to store calculated values
pe_node_allocation = []

policies = ['fixed', 'random', 'random-exclusion', 'residual']
policies = ['random']

print(" -------- Iterating Through Designs -------- ")

# Iterate through each design
for design in designs:
    # Calculate performance value
    number_pes = design['design']['number_pes'] 
    nodes_updt_per_pe = design['design']['nodes_updt_per_pe']
    cache = design['design']['cache']
    policy = design['design']['policy']
    latency = design['latency']['latency_total']
    n_p = number_pes * nodes_updt_per_pe
    Rcf = n_p/latency
    if {"number_pes": number_pes, "nodes_updt_per_pe": nodes_updt_per_pe, "np": n_p, "Rcf": Rcf, "cache": cache,"policy": policy} not in pe_node_allocation:
        pe_node_allocation.append({"number_pes": number_pes, "nodes_updt_per_pe": nodes_updt_per_pe, "np": n_p, "Rcf":Rcf, "cache": cache, "policy": policy})

"""Run Simulations"""

# Option 1: Suppress all warnings
warnings.filterwarnings("ignore")

set_plot_options()
colors = get_plot_colors()

data_gen = DataGenerator()
result_analyzer = AnalyzeResult()

print(" -------- Fetching Data -------- ")

# fetch 
# file_path = 'GBP_Simulations/GBP/Raw_Datasets/data/input_MITb_g2o.g2o'
# data_gen.generate_SLAM_dataset(file_path=file_path)

# dataset = 'input_MITb_g2o'
# filepath_n = 'GBP_Simulations/GBP/Raw_Datasets/gbp_data'
# factor_data = os.path.join(filepath_n, f'{dataset}_factor_data.txt')
# marginal_data = os.path.join(filepath_n, f'{dataset}_marginal_data.txt')
# A, b = data_gen.fetch_SLAM_dataset(file_path_factor=factor_data, file_path_marginal=marginal_data)

A, b = data_gen.get_1D_line_matrix(int(graph['N']), scaling=True, normalized=False)
graph = NetworkxGraph(A)

# sync convergence
sync_convergence_threshold = 1*10**-5 # convergence threshold
convergence_type = 'hi_x' #all
simulation_convergence_threshold = 1*10**-2

# Initialize lists to store number of steam passes
pe_node_convergence = []

# Run syncrhonous implementation
num_iterations = 100
for it in range(0,num_iterations):
    P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_divs, means, iteration = run_GaBP_SYNC_ACCELERATED(A, 
                                                                                                      b, 
                                                                                                      max_iter=100_000, 
                                                                                                      mae=True if convergence_type == 'mae' else False, 
                                                                                                      convergence_threshold=simulation_convergence_threshold,
                                                                                                      show=True)
    final_mean = list(mu_i)
    if iteration < 10000:
        print("True Marginals Computed")
        break
    else:
        iteration = float("inf")
        print("=========== RESTART ===========")

# iterate over PE and node updates per PE
num_iterations = 10
total_iterations = len(pe_node_allocation) * num_iterations
pbar = tqdm(total=total_iterations, desc="Running GBP Simulations", unit="designs")

manager = multiprocessing.Manager()
output_list = manager.list()  # Create a shared lis

# Define the number of threads you want to use
num_threads = 6  # You can adjust this based on your system's capabilities

# Calculate chunk size and distribute the remainder
chunk_size = len(pe_node_allocation) // num_threads
remainder = len(pe_node_allocation) % num_threads
# Create chunks with adjusted sizes
design_chunks = []
start = 0
for i in range(num_threads):
    chunk_end = start + chunk_size + (1 if i < remainder else 0)
    design_chunks.append(pe_node_allocation[start:chunk_end])
    start = chunk_end

progress_queue = multiprocessing.Queue()

# Create and start processes
processes = []
for chunk in design_chunks:
    process = multiprocessing.Process(target=process_designs, args=(chunk, num_iterations, progress_queue, output_list))
    processes.append(process)
    process.start()

progress_thread = threading.Thread(target=update_progress, args=(progress_queue, total_iterations))
progress_thread.start()

# Wait for all processes to complete
for process in processes:
    process.join()

progress_queue.close()
progress_thread.join()

output_designs = list(output_list)
    
filename_to_save_design = f"Hardware_Model/simulations/simulations_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json" if filename == "" else f"Hardware_Model/simulations/{str(filename)}.json"

# Write the list of dictionaries to the file
with open(filename_to_save_design, 'w') as file:
    json.dump(output_designs, file, indent=4)