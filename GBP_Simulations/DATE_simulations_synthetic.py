from GBP.data import DataGenerator
from GBP.gbp import run_GaBP_SYNC_ACCELERATED, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL_T_nP, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL_T_MAX, run_GaBP_HARDWARE_ACCELERATED, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL_NEW, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_STREAMS, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_STREAMS_STOCHASTIC
# , run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_2_STREAMS
from GBP.utilities import HiddenPrints
from GBP.visulisation import set_plot_options, get_plot_colors, NetworkxGraph, AnalyzeResult
 
import warnings
import matplotlib
import numpy as np
import math
import random
import time
import pandas as pd

from datetime import datetime

""" ------------------------- HELPER FUNCTIONS ------------------------- """
def unique_np_pairs(filename, design_name):
    
    # Load the existing design_summary.csv file
    df = pd.read_csv(design_name)

    # Filter the rows for the filename 'line_1000_dynamic'
    filtered_df = df[df['Filename'] == filename]

    # Create a set of unique (n, p) pairs to remove duplicates
    np_set = {(row['n'], row['p']) for _, row in filtered_df.iterrows()}

    # Convert the set of tuples back to a list of dictionaries
    np_pairs_unique = [{'n': n, 'p': p, 'iterations': [], 'iterations_ave': -1} for n, p in np_set]

    return np_pairs_unique

def extract_and_write_data(filename, design_name, dictionary):
    
    # Load the existing design_summary.csv file
    df = pd.read_csv(design_name)

    # Filter the rows for the specific filename
    filtered_df = df[df['Filename'] == filename]

    # Initialize 'iterations', 'iterations_ave', and 'total_latency' columns if they don't exist
    if 'iterations' not in filtered_df.columns:
        filtered_df['iterations'] = np.nan

    if 'iterations_ave' not in filtered_df.columns:
        filtered_df['iterations_ave'] = np.nan

    if 'total_latency' not in filtered_df.columns:
        filtered_df['total_latency'] = np.nan

    # Iterate over np pairs in the provided dictionary and write iterations and average iterations
    for D in dictionary:
        n, p, iterations, iterations_ave = D['n'], D['p'], D['iterations'], D['iterations_ave']

        # Write iterations only if the row matches n and p, and the value is NaN (doesn't exist yet)
        filtered_df['iterations'] = filtered_df.apply(
            lambda row: iterations if row['n'] == n and row['p'] == p and pd.isna(row['iterations']) else row['iterations'],
            axis=1
        )

        # Write iterations_ave only if the row matches n and p, and the value is NaN (doesn't exist yet)
        filtered_df['iterations_ave'] = filtered_df.apply(
            lambda row: iterations_ave if row['n'] == n and row['p'] == p and pd.isna(row['iterations_ave']) else row['iterations_ave'],
            axis=1
        )

    # Calculate 'total_latency' as 'stream_latency' * 'iterations_ave'
    # Ensure 'stream_latency' exists or create it if not
    if 'stream_latency' not in filtered_df.columns:
        filtered_df['stream_latency'] = np.nan

    filtered_df['total_latency'] = filtered_df.apply(
        lambda row: row['stream_latency'] * row['iterations_ave'] if pd.notna(row['stream_latency']) and pd.notna(row['iterations_ave']) else np.nan,
        axis=1
    )

    # Generate the current date and time string in the format day_month_hour_min
    current_time = datetime.now().strftime("%d_%m_%H_%M")

    # Save the filtered data with the updated columns to a CSV file
    filtered_df.to_csv(f'DATE_2024/{filename}_{current_time}.csv', index=False)

    return


""" ------------------------- CLASS INPUTS ------------------------- """

# Option 1: Suppress all warnings
warnings.filterwarnings("ignore")

set_plot_options()
colors = get_plot_colors()

data_gen = DataGenerator()
result_analyzer = AnalyzeResult()

""" ------------------------- SIMULATION SETUP ------------------------- """

# Number of nodes
num_nodes = 10000
sync_convergence_threshold = 1*10**-8
async_convergence_threshold = 1*10**-5

# starting mae for convergence
starting_mae_for_simulation = 10

# total number of simulations
number_simulations = 25
num_iterations = 10

# filename
# filename_baseline = 'line_1000_static'
# filename_residual = 'line_1000_dynamic'

# filename_baseline = 'lattice_1000_static'
# filename_residual = 'lattice_1000_dynamic'

# filename_baseline = 'line_10000_static'
# filename_residual = 'line_10000_dynamic'

filename_baseline = 'lattice_10000_static'
filename_residual = 'lattice_10000_dynamic'

print(f"================== {filename_baseline} ==================")
print(f"================== {filename_residual} ==================")

# unique np pairs
baseline_np_dicts = unique_np_pairs(filename_baseline, 'design_summary_static.csv')
residual_np_dicts = unique_np_pairs(filename_residual, 'design_summary_dynamic.csv')

" -------------------------- REPEAT SIMULATIONS AND COLLECT RESULTS --------------------------"

start = time.time()

for ns in range(number_simulations):
    
    # ------------------- setting up simulations -------------------
    print(f"-------------- GENERATING GBP MATRIX: {ns+1} --------------")
    sum_of_iterations = 0
    for _ in range(1000):
        # A,b = data_gen.get_1D_line_matrix_PSD_difficult(num_nodes, eigenvalue_spread=1e-5, regularization_strength=1e-2, noise_strength=1e-2)
        # A,b = data_gen.get_2D_lattice_matrix_PSD_difficult(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)+1), eigenvalue_spread=1e-5, regularization_strength=1e-2, noise_strength=1e-2)

        A,b = data_gen.get_2D_lattice_matrix_PSD_difficult(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)), eigenvalue_spread=1e-5, regularization_strength=1e-2, noise_strength=1e-2)
        graph = NetworkxGraph(A)
        P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_divs, means, iteration = run_GaBP_SYNC_ACCELERATED(A, b, max_iter=1000, mae=False, convergence_threshold=sync_convergence_threshold, show=False)
        sum_of_iterations += iteration
        final_mean = list(mu_i)
        final_std = P_i
        if iteration < 1000:
            break
        else:
            print("error: restarting gbp matrix creation")
    P_i, mu_i, iteration, STARTING_MAE = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=1, number_pes=1, TRUE_MEAN=final_mean, max_iter=1, mae=True, convergence_threshold=async_convergence_threshold, show=False)
    print(f"-------------- SUCCESSFULLY BUILT GBP MATRIX: {ns+1} --------------")

    # ------------------- iterate over all np pairs in order to compute latencies -------------------
        
    # ------------------- baseline -------------------
    print(f" ============== BASELINE = {ns+1} ============== ")
    for D_BASELINE in baseline_np_dicts:
        sum_of_iterations = 0
        for it in range(0,num_iterations):
            x = True
            while x:
                P_i, mu_i, iteration, _ = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=D_BASELINE['n'], number_pes=D_BASELINE['p'], TRUE_MEAN=final_mean, max_iter=1500, mae=True, convergence_threshold=async_convergence_threshold, show=False)
                if iteration < 1500-1:
                    x = False
            sum_of_iterations += iteration
        ave = sum_of_iterations/num_iterations
        D_BASELINE['iterations'].append(ave)
        print(f"(n={D_BASELINE['n']},p={D_BASELINE['p']}) => Streams = {ave}")
        
    # ------------------- residual -------------------
    print(f" ============== RESIDUAL = {ns+1} ============== ")
    for D_RESIDUAL in residual_np_dicts:
        sum_of_iterations = 0
        for it in range(0,num_iterations):
            x = True
            while x:
                P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED_RESIDUAL(A, b, caching=True, node_updates_per_pe=D_RESIDUAL['n'], number_pes=D_RESIDUAL['p'], TRUE_MEAN=final_mean, max_iter=1500, mae=True, convergence_threshold=async_convergence_threshold, show=False)
                if iteration < 1500-1:
                    x = False
            sum_of_iterations += iteration
        ave = sum_of_iterations/num_iterations
        D_RESIDUAL['iterations'].append(ave)
        print(f"(n={D_RESIDUAL['n']},p={D_RESIDUAL['p']}) => Streams = {ave}")

print("------ BASELINE ------")
for D in baseline_np_dicts:
    D['iterations_ave'] = sum(D['iterations'])/len(D['iterations'])
print(baseline_np_dicts)
extract_and_write_data(filename_baseline, 'design_summary_static.csv', baseline_np_dicts)
print("------ RESIDUAL ------")
for D_RESIDUAL in residual_np_dicts:
    D_RESIDUAL['iterations_ave'] = sum(D_RESIDUAL['iterations'])/len(D_RESIDUAL['iterations'])
print(residual_np_dicts)
extract_and_write_data(filename_residual, 'design_summary_dynamic.csv', residual_np_dicts)

end = time.time()

print(f"TOTAL TIME ELAPSED = {end - start} seconds")
print(f"TOTAL TIME ELAPSED = {(end - start) / 60} minutes")