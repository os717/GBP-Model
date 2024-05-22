from Hardware_Model.components.compute_unit import *
from Hardware_Model.components.control_unit import *

import math

class Latency:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _data_setup(nodes_updt_per_pe, number_pes=1):
        setup_latency = 1 + 2 * number_pes * nodes_updt_per_pe
        return setup_latency
    
    @staticmethod
    def _data_factors(num_factors):
        factor_time = 1 + 11 * num_factors
        return factor_time

    @staticmethod
    def _data_marginals(graph_size):
        marginal_time = 1 + 13 * graph_size
        return marginal_time

    @staticmethod
    def _stream1_latency(nodes_updt_per_pe, num_factors, number_pes):
        latency_setup = Latency._data_setup(nodes_updt_per_pe, number_pes) 
        latency_factors = Latency._data_factors(num_factors)
        return latency_setup + latency_factors
    
    @staticmethod
    def _stream2_latency(graph_size):
        latency_marginals = Latency._data_marginals(graph_size)
        return latency_marginals 

    @staticmethod
    def _compute_latency(nodes_updt_per_pe, ave_valency, compute_unroll_factor):

        # time for inverse
        time_inverse = inv_mtx[str(compute_unroll_factor['inv'])]['latency']

        # time for mult
        time_mult_mtx_mtx = mult_mtx_vec[str(compute_unroll_factor['mul'])]['latency']
        time_mult_total = time_mult_mtx_mtx * 2  # reuse the matrix block

        # time for artihmetic (add-accumulate & sub)
        time_arith_mtx = arith_mtx[str(compute_unroll_factor['add'])]['latency']
        time_arith_total = time_arith_mtx * 2  # reuse the arithmetic block

        # time for one pass compute block
        time_compute_block = time_inverse + time_mult_total + time_arith_total + time_arith_total

        # time for compute
        time_compute = time_inverse * ave_valency * nodes_updt_per_pe + time_compute_block

        # time for searching factors
        time_search_factors = math.ceil(nodes_updt_per_pe * nodes_updt_per_pe * ave_valency) 

        # time for searching marginals
        time_search_marginals = math.ceil(nodes_updt_per_pe * nodes_updt_per_pe * (ave_valency + 1) / 2 * ave_valency)

        return max(time_search_factors, time_search_marginals, time_compute)
    
    @staticmethod
    def total_latency(graph_size, num_factors, fpga_clk, memory_bw, nodes_updt_per_pe, number_pes, compute_unroll_factor, double_buffering, capping):
        latency_s1 = Latency._stream1_latency(min(graph_size, nodes_updt_per_pe), num_factors, number_pes) * memory_bw
        latency_s2 = Latency._stream2_latency(graph_size) * memory_bw
        ave_valency = num_factors / graph_size if capping == None else capping
        latency_compute = Latency._compute_latency(min(graph_size, nodes_updt_per_pe), ave_valency, compute_unroll_factor) * fpga_clk

        stream_latencies = {
            'latency_compute'         : latency_compute,
            'latency_stream1'         : latency_s1,
            'latency_stream2'         : latency_s2,
            'latency_total'           : latency_s2 + max(latency_s1, latency_compute) if double_buffering else latency_s1  + latency_s2  + latency_compute 
        }

        return stream_latencies