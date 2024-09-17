from GBP.data import DataGenerator
from GBP.gbp import run_GaBP_SYNC_ACCELERATED, run_GaBP_HARDWARE_BESTCASE_RESIDUAL, run_GaBP_HARDWARE_BESTCASE, run_GaBP_HARDWARE_ACCELERATED, run_GaBP_HARDWARE_ACCELERATED_RESIDUAL, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_STREAMS, run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_STREAMS_STOCHASTIC
# , run_GaBP_HARDWARE_ACCELERATED_EXCLUSION_NEIGHBOURS_ACROSS_2_STREAMS
from GBP.utilities import HiddenPrints
from GBP.visulisation import set_plot_options, get_plot_colors, NetworkxGraph, AnalyzeResult
 
import warnings
import matplotlib
import numpy as np
import math
import random

# Option 1: Suppress all warnings
warnings.filterwarnings("ignore")

set_plot_options()
colors = get_plot_colors()

data_gen = DataGenerator()
result_analyzer = AnalyzeResult()

num_nodes = 1000

sync_convergence_threshold = 1*10**-8

# NODE_UPDT_PE = 137
# PEs = 5

sum_of_iterations = 0
num_iterations = 1

for _ in range(1000):
    # A, b = data_gen.get_1D_line_matrix_PSD(num_nodes)
    # A, b = data_gen.get_1D_line_matrix(num_nodes, scaling=False, normalized=False)
    A,b = data_gen.get_2D_lattice_matrix_PSD(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)))
    # A,b = data_gen.get_2D_lattice_matrix(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)))

    # A,b = data_gen.get_2D_lattice_matrix_PSD_shaped(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)), eigenvalue_spread=10)

    # b = 10 * b / 0.00874062402639083
    
    graph = NetworkxGraph(A)
    P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist, stand_divs, means, iteration = run_GaBP_SYNC_ACCELERATED(A, b, max_iter=10000, mae=False, convergence_threshold=sync_convergence_threshold, show=True)
    sum_of_iterations += iteration
    final_mean = list(mu_i)
    final_std = P_i
    if iteration < 10000:
        break
    else:
        print("=========== RESTART ===========")

print(f"AVE. SYNC ITERATIONS = {sum_of_iterations/num_iterations}")

async_convergence_threshold = 1*10**-5

sum_of_iterations = 0
num_iterations = 1


NODE_UPDT_PE = 84
PEs = 5

for it in range(0,num_iterations):
    # print(f"-------------- ITERATION = {it+1} --------------")
    P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=NODE_UPDT_PE, number_pes=PEs, TRUE_MEAN=final_mean, max_iter=1000, mae=True, convergence_threshold=async_convergence_threshold, show=True)
    sum_of_iterations += iteration
    print(f"-------------- ITERATION = {it+1} => Streams = {iteration} --------------")

print(f"AVE. ASYNC ITERATIONS = {sum_of_iterations/num_iterations}")


