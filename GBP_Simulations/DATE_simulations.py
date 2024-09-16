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


# Option 1: Suppress all warnings
warnings.filterwarnings("ignore")

set_plot_options()
colors = get_plot_colors()

data_gen = DataGenerator()
result_analyzer = AnalyzeResult()

""" ------------------------- 1 ITERATION OF THE GBP ------------------------- """

start = time.time()

# Number of nodes
num_nodes = 10000
sync_convergence_threshold = 1*10**-8
async_convergence_threshold = 1*10**-5

# starting mae for convergence
starting_mae_for_simulation = 10

# updates
BASELINE = []
# RESIDUAL_HW = []
RESIDUAL_IDEAL = []

# total number of simulations
number_simulations = 25

" -------------------------- REPEAT SIMULATIONS AND COLLECT RESULTS --------------------------"
for ns in range(number_simulations):

    # ------------- RUN FIRST TIME -------------
    print(f"-------------- GENERATING GBP MATRIX: {ns+1} --------------")
    num_iterations = 10
    sum_of_iterations = 0
    for _ in range(1000):
        
        # A,b = data_gen.get_2D_lattice_matrix_PSD_difficult(int(math.sqrt(num_nodes)), int(math.sqrt(num_nodes)), eigenvalue_spread=1e-5, regularization_strength=1e-2, noise_strength=1e-2)

        A,b = data_gen.get_1D_line_matrix_PSD_difficult(num_nodes, eigenvalue_spread=1e-5, regularization_strength=1e-2, noise_strength=1e-2)

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
    # -------------------------- RUN SIMULATIONS --------------------------

    # --------- BASELINE --------- 
    sum_of_iterations = 0
    num_iterations = 10
    NODE_UPDT_PE = 84
    PEs = 5
    for it in range(0,num_iterations):
        x = True
        while x:
            P_i, mu_i, iteration, _ = run_GaBP_HARDWARE_ACCELERATED(A, b, caching=False, node_updates_per_pe=NODE_UPDT_PE, number_pes=PEs, TRUE_MEAN=final_mean, max_iter=1000, mae=True, convergence_threshold=async_convergence_threshold, show=False)
            if iteration < 1000-1:
                x = False
        sum_of_iterations += iteration
    BASELINE.append(sum_of_iterations/num_iterations)
    print(f"BASELINE = {ns+1} => Streams = {sum_of_iterations/num_iterations}")


    # # --------- RESIDUAL HW ---------
    # sum_of_iterations = 0
    # num_iterations = 10
    # NODE_UPDT_PE = 84
    # PEs = 4
    # for it in range(0,num_iterations):
    #     x = True
    #     while x:
    #         P_i, mu_i, iteration, _, _ = run_GaBP_HARDWARE_ACCELERATED_RESIDUAL_T_MAX(A, b, caching=True, node_updates_per_pe=NODE_UPDT_PE, number_pes=PEs, TRUE_MEAN=final_mean, max_iter=1000, mae=True, convergence_threshold=async_convergence_threshold, show=False)
    #         if iteration < 1000-1:
    #             x = False
    #     sum_of_iterations += iteration
    # RESIDUAL_HW.append(sum_of_iterations/num_iterations)
    # print(f"RESIDUAL_HW = {ns+1} => Streams = {sum_of_iterations/num_iterations}")

    # # --------- RESIDUAL IDEAL ---------
    sum_of_iterations = 0
    num_iterations = 3
    NODE_UPDT_PE = 84
    PEs = 4
    for it in range(0,num_iterations):
        x = True
        while iteration > 1000-1:
            P_i, mu_i, iteration = run_GaBP_HARDWARE_ACCELERATED_RESIDUAL(A, b, caching=True, node_updates_per_pe=NODE_UPDT_PE, number_pes=PEs, TRUE_MEAN=final_mean, max_iter=1000, mae=True, convergence_threshold=async_convergence_threshold, show=False)
            if iteration < 1000-1:
                x = False
        sum_of_iterations += iteration
    RESIDUAL_IDEAL.append(sum_of_iterations/num_iterations)
    print(f"RESIDUAL_IDEAL = {ns+1} => Streams = {sum_of_iterations/num_iterations}")

print(f"BASELINE = {BASELINE} => {sum(BASELINE)/len(BASELINE)}")
# print(f"RESIDUAL_HW = {RESIDUAL_HW} => {sum(RESIDUAL_HW)/len(RESIDUAL_HW)}")
print(f"RESIDUAL_IDEAL = {RESIDUAL_IDEAL} => {sum(RESIDUAL_IDEAL)/len(RESIDUAL_IDEAL)}")


end = time.time()

print(f"TOTAL TIME ELAPSED = {end - start} seconds")
print(f"TOTAL TIME ELAPSED = {(end - start) / 60} minutes")