import logging
from datetime import datetime
import pandas as pd
import numpy as np
import math
import time
import warnings

from GBP.data import DataGenerator
from GBP.gbp import (
    run_GaBP_SYNC_ACCELERATED,
    run_GaBP_HARDWARE_ACCELERATED_RESIDUAL,
    run_GaBP_HARDWARE_ACCELERATED
)
from GBP.visulisation import set_plot_options, get_plot_colors, NetworkxGraph, AnalyzeResult

# Generate a log file name based on the current time
log_filename = datetime.now().strftime("logfile_%Y-%m-%d_%H-%M-%S.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()             # Log to console
    ]
)

logging.info("Script started at %s", datetime.now())

""" ------------------------- HELPER FUNCTIONS ------------------------- """
def unique_np_pairs(filename, design_name):
    logging.info("Extracting unique (n, p) pairs from %s", design_name)
    
    df = pd.read_csv(design_name)
    filtered_df = df[df['Filename'] == filename]
    np_set = {(row['n'], row['p']) for _, row in filtered_df.iterrows()}
    np_pairs_unique = [{'n': n, 'p': p, 'iterations': [], 'iterations_ave': -1} for n, p in np_set]

    return np_pairs_unique

def extract_and_write_data(filename, design_name, dictionary):
    logging.info("Extracting data from %s and updating it for %s", design_name, filename)
    
    df = pd.read_csv(design_name)
    filtered_df = df[df['Filename'] == filename]

    if 'iterations' not in filtered_df.columns:
        filtered_df['iterations'] = np.nan
    if 'iterations_ave' not in filtered_df.columns:
        filtered_df['iterations_ave'] = np.nan
    if 'total_latency' not in filtered_df.columns:
        filtered_df['total_latency'] = np.nan

    for D in dictionary:
        n, p, iterations, iterations_ave = D['n'], D['p'], D['iterations'], D['iterations_ave']

        filtered_df['iterations'] = filtered_df.apply(
            lambda row: iterations if row['n'] == n and row['p'] == p and pd.isna(row['iterations']) else row['iterations'],
            axis=1
        )
        filtered_df['iterations_ave'] = filtered_df.apply(
            lambda row: iterations_ave if row['n'] == n and row['p'] == p and pd.isna(row['iterations_ave']) else row['iterations_ave'],
            axis=1
        )

    if 'stream_latency' not in filtered_df.columns:
        filtered_df['stream_latency'] = np.nan

    filtered_df['total_latency'] = filtered_df.apply(
        lambda row: row['stream_latency'] * row['iterations_ave'] if pd.notna(row['stream_latency']) and pd.notna(row['iterations_ave']) else np.nan,
        axis=1
    )

    current_time = datetime.now().strftime("%d_%m_%H_%M_%S")
    output_filename = f'DATE_2024/{filename}_{current_time}.csv'
    filtered_df.to_csv(output_filename, index=False)

    logging.info("Data written to %s", output_filename)

""" ------------------------- CLASS INPUTS ------------------------- """

warnings.filterwarnings("ignore")
set_plot_options()
colors = get_plot_colors()

data_gen = DataGenerator()
result_analyzer = AnalyzeResult()

""" ------------------------- SIMULATION SETUP ------------------------- """

num_nodes = 10000
sync_convergence_threshold = 1 * 10**-8
async_convergence_threshold = 1 * 10**-5
starting_mae_for_simulation = 10
number_simulations = 1
num_iterations = 1

# filename
# filename_baseline = 'line_1000_static'
# filename_residual = 'line_1000_dynamic'

# filename_baseline = 'lattice_1000_static'
# filename_residual = 'lattice_1000_dynamic'

# filename_baseline = 'line_10000_static'
# filename_residual = 'line_10000_dynamic'

filename_baseline = 'lattice_10000_static'
filename_residual = 'lattice_10000_dynamic'

logging.info("Baseline filename: %s", filename_baseline)
logging.info("Residual filename: %s", filename_residual)

baseline_np_dicts = unique_np_pairs(filename_baseline, 'design_summary_static.csv')
residual_np_dicts = unique_np_pairs(filename_residual, 'design_summary_dynamic.csv')

""" ------------------------- SIMULATION ------------------------- """


en_spread = 1e-1
logging.info("Eigenvalue Spread {}".format(en_spread))


start = time.time()

for ns in range(number_simulations):
    logging.info("Starting simulation #%d", ns+1)

    sum_of_iterations = 0
    for _ in range(1000):
        # A,b = data_gen.get_1D_line_matrix_PSD_difficult(num_nodes, eigenvalue_spread=en_spread, regularization_strength=1e-2, noise_strength=1e-2)
        # A, b = data_gen.get_2D_lattice_matrix_PSD_difficult(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)+1), eigenvalue_spread=en_spread, regularization_strength=1e-2, noise_strength=1e-2)
        A, b = data_gen.get_2D_lattice_matrix_PSD_difficult(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)), eigenvalue_spread=en_spread, regularization_strength=1e-2, noise_strength=1e-2)
        graph = NetworkxGraph(A)
        P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_divs, means, iteration = run_GaBP_SYNC_ACCELERATED(A, b, max_iter=300, mae=False, convergence_threshold=sync_convergence_threshold, show=True)
        sum_of_iterations += iteration
        final_mean = list(mu_i)
        if iteration < 300:
            break
        else:
            logging.error("Error: Restarting GBP matrix creation")

    P_i, mu_i, iteration, STARTING_MAE = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=1, number_pes=1, TRUE_MEAN=final_mean, max_iter=1, mae=True, convergence_threshold=async_convergence_threshold, show=False)
    logging.info("Successfully built GBP matrix for simulation #%d", ns+1)

    # Baseline
    for D_BASELINE in baseline_np_dicts:
        sum_of_iterations = 0
        failed = 0
        for it in range(num_iterations):
            count = 0
            while True:
                P_i, mu_i, iteration, _ = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=D_BASELINE['n'], number_pes=D_BASELINE['p'], TRUE_MEAN=final_mean, max_iter=2500, mae=True, convergence_threshold=async_convergence_threshold, show=True)
                if iteration < 2500-1:
                    logging.info("Baseline: lattice(n=%d, p=%d) iterations = %d", D_BASELINE['n'], D_BASELINE['p'], iteration)
                    break
                else:
                    count += 1
                    logging.warning("Baseline iteration count %d", count)
                if count > 3:
                    logging.error("Baseline failed after 5 attempts")
                    failed += 1
                    break
            if count <= 3:
                sum_of_iterations += iteration
        ave = sum_of_iterations / (num_iterations - failed) if num_iterations - failed != 0 else 0
        logging.info("Baseline: lattice(n=%d, p=%d) AVERAGE = %d", D_BASELINE['n'], D_BASELINE['p'], ave)
        logging.info("====================")
        D_BASELINE['iterations'].append(ave)

    # Residual
    for D_RESIDUAL in residual_np_dicts:
        sum_of_iterations = 0
        failed = 0
        for it in range(num_iterations):
            count = 0
            while True:
                P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED_RESIDUAL(A, b, caching=True, node_updates_per_pe=D_RESIDUAL['n'], number_pes=D_RESIDUAL['p'], TRUE_MEAN=final_mean, max_iter=2500, mae=True, convergence_threshold=async_convergence_threshold, show=True)
                if iteration < 2500-1:
                    logging.info("Residual: lattice(n=%d, p=%d) iterations = %d", D_RESIDUAL['n'], D_RESIDUAL['p'], iteration)
                    break
                else:
                    count += 1
                    logging.warning("Residual iteration count %d", count)
                if count > 3:
                    logging.error("Residual failed after 5 attempts")
                    failed += 1
                    break
            if count <= 3:
                sum_of_iterations += iteration
        ave = sum_of_iterations / (num_iterations - failed) if num_iterations - failed != 0 else 0
        logging.info("Residual: lattice(n=%d, p=%d) AVERAGE = %d", D_BASELINE['n'], D_BASELINE['p'], ave)
        logging.info("====================")
        D_RESIDUAL['iterations'].append(ave)



logging.info("------ BASELINE ------")
for D in baseline_np_dicts:
    D['iterations_ave'] = sum(D['iterations'])/len(D['iterations'])
logging.info(baseline_np_dicts)
extract_and_write_data(filename_baseline, 'design_summary_static.csv', baseline_np_dicts)

logging.info("------ RESIDUAL ------")
for D_RESIDUAL in residual_np_dicts:
    D_RESIDUAL['iterations_ave'] = sum(D_RESIDUAL['iterations'])/len(D_RESIDUAL['iterations'])
logging.info(residual_np_dicts)
extract_and_write_data(filename_residual, 'design_summary_dynamic.csv', residual_np_dicts)

end = time.time()
elapsed_time = end - start

logging.info("Total time elapsed: %f seconds", elapsed_time)
logging.info("Total time elapsed: %f minutes", elapsed_time / 60)