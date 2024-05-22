from Hardware_Model.library.latency import Latency
from Hardware_Model.library.pe_resources import PE_Resources
from Hardware_Model.library.system_resources import System_Resources

class Hardware_Model():
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate(
            # graph topology
            graph_size, num_factors, 
            
            # fpga specifications
            fpga_resources, fpga_clk, memory_bw,

            # hardware specifications
            nodes_updt_per_pe, number_pes, compute_unroll_factor,

            # double buffering
            double_buffering,

            # resource scaling for the binary search
            binary_search_scaling = 1,

            # capping
            capping = None

            ):

        ave_valency = num_factors/graph_size

        # total latency and design resources
        system = System_Resources.total_system_resources(fpga_resources, nodes_updt_per_pe, number_pes, ave_valency, compute_unroll_factor, binary_search_scaling, double_buffering, capping)
        system['latency'] = Latency.total_latency(graph_size, num_factors, fpga_clk, memory_bw, nodes_updt_per_pe, number_pes, compute_unroll_factor, double_buffering, capping)

        return system