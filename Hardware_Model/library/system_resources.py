from Hardware_Model.components.compute_unit import *
from Hardware_Model.components.control_unit import *

from Hardware_Model.library.pe_resources import PE_Resources

import math

class System_Resources:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _instantiated_PEs(resources_total, resources_single, dma_overhead):
        available_lut = (resources_total['lut'] - dma_overhead['distributed']['lut']) 
        available_ff = (resources_total['ff'] - dma_overhead['distributed']['ff']) 
        available_bram = (resources_total['bram'] - dma_overhead['distributed']['bram']) 
        available_dsp = (resources_total['dsp'] - dma_overhead['distributed']['dsp']) 

        # calculate max number of PEs from total resources available
        max_lut = int(available_lut / resources_single['lut'])
        max_ff = int(available_ff / resources_single['ff'])
        max_bram = int(available_bram / resources_single['bram'])
        max_dsp = int(available_dsp / resources_single['dsp'])

        return min(max_lut, max_ff, max_bram, max_dsp)

    @staticmethod
    def total_system_resources(fpga_resources, nodes_updt_per_pe, number_pes, ave_valency, unroll_factor, placeback, double_buffering, capping=None):
        pe_resources = PE_Resources.total_PE_resources(nodes_updt_per_pe, 
                                                       ave_valency if capping==None else min(ave_valency, capping), 
                                                       unroll_factor, 
                                                       placeback, 
                                                       double_buffering)
                
        resources_total = {key: value * number_pes for key, value in pe_resources.items()}

        resources_total['lut'] += dma_overhead['distributed']['lut']
        resources_total['ff'] += dma_overhead['distributed']['ff']
        resources_total['bram'] += dma_overhead['distributed']['bram']
        resources_total['dsp'] += dma_overhead['distributed']['dsp']

        resources_percent = {'lut': 100*resources_total['lut']/fpga_resources['lut'], 
                             'ff': 100*resources_total['ff']/fpga_resources['ff'], 
                             'bram': 100*resources_total['bram']/fpga_resources['bram'], 
                             'dsp': 100*resources_total['dsp']/fpga_resources['dsp']}

        resources_dict = {'resources_total': resources_total, 
                          'resources_pe': pe_resources, 
                          'resources_%': resources_percent}
        
        design_dict = {'number_pes': number_pes, 
                       'nodes_updt_per_pe': nodes_updt_per_pe, 
                       'nodes_updt_per_stream': number_pes*nodes_updt_per_pe, 
                       'compute_unroll_factors': unroll_factor, 
                       'binary_searcher': {'buffer_size': math.ceil(math.log2(nodes_updt_per_pe)) + 2, 
                                           'resource_scaling': placeback}, 
                       'double_buffering': double_buffering, 
                       'capping': capping}

        return {'design': design_dict, 'resources': resources_dict}