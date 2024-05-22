from Hardware_Model.components.compute_unit import *
from Hardware_Model.components.control_unit import *

import math

class PE_Resources:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _fetch_parameterisation(inp_param, dict_name):
        keys = sorted(dict_name.keys(), key=int)
        for key in keys:
            if int(key) >= int(inp_param):
                return dict_name[key]
        return None

    @staticmethod
    def _num_BRAM(data_amount):
        capacity = 36 * 1024  # 36 kilobits
        return math.ceil(data_amount / capacity)

    @staticmethod
    def _memory_PE_resources(nodes_updt, ave_valency, double_buffering):
        # create member variables
        resources = 0
        pe_bram_dict = {}

        # amount of data required to be stored in bram
        ram_amount = (
                # number of rams
                32 * 1 * nodes_updt * 1 +
                32 * 1 * nodes_updt * ave_valency +
                32 * 9 * nodes_updt * ave_valency +
                32 * 13 * nodes_updt * 1 +
                32 * 13 * nodes_updt * ave_valency
        )
        
        if double_buffering:
            ram_amount *= 2

        fifo_data = (
                # number of fifo
                32 * 1 * nodes_updt * ave_valency +
                32 * 9 * nodes_updt * ave_valency +
                32 * 11 * 2 +
                32 * 12 * nodes_updt * ave_valency +
                32 * 12 * nodes_updt * 1 +
                32 * 13 * 2
        )

        resources = PE_Resources._num_BRAM(ram_amount + fifo_data)

        # return dictionary
        pe_bram_dict = {
            'lut': 0,
            'ff': 0,
            'bram': resources,
            'dsp': 0
        }
        return pe_bram_dict

    @staticmethod
    def _compute_PE_resources(unroll_factor_list):
        resources = {}
        for key in ['lut', 'ff', 'bram', 'dsp']:
            resources[key] = (
                    PE_Resources._fetch_parameterisation(str(math.floor(unroll_factor_list['inv'])), inv_mtx)[key] +
                    PE_Resources._fetch_parameterisation(str(math.floor(unroll_factor_list['mul'])), mult_mtx_mtx)[key] +
                    PE_Resources._fetch_parameterisation(str(math.floor(unroll_factor_list['add'])), arith_mtx)[key] +
                    PE_Resources._fetch_parameterisation(str(math.floor(unroll_factor_list['sub'])), arith_mtx)[key]
            )
        return resources

    @staticmethod
    def _control_PE_resources(inp_nodes_updt, inp_ave_valency, placeback):
        resources = {}
        for key in ['lut', 'ff', 'bram', 'dsp']:
            resources[key] = (
                    PE_Resources._fetch_parameterisation(str(math.ceil(math.log2(inp_nodes_updt))), pe_core)[key] +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt)), parser)[key] +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt / (2 * placeback))),
                                                          binary_search)[key] +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt * inp_ave_valency)),
                                                          index_sorting)[key] +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt + (inp_nodes_updt * inp_ave_valency) / 2)),
                                                          consumer)[key] * 2 +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt)), search_marginals)[key] +
                    PE_Resources._fetch_parameterisation(str(math.ceil(inp_nodes_updt * inp_ave_valency)),
                                                          search_factors)[key]
            )
        return resources

    @staticmethod
    def total_PE_resources(inp_nodes_updt, inp_ave_valency, inp_unroll_factor, placeback, double_buffering, capping=None):
        # total resources
        total_resources = {}

        # total memory unit overhead
        memory_overhead = PE_Resources._memory_PE_resources(inp_nodes_updt, inp_ave_valency, double_buffering)

        # total control unit overhead
        control_overhead = PE_Resources._control_PE_resources(inp_nodes_updt, inp_ave_valency, placeback)

        # total compute unit overhead
        compute_overhead = PE_Resources._compute_PE_resources(inp_unroll_factor)

        # fetch total PE resources
        for key in ['lut', 'ff', 'bram', 'dsp']:
            total_resources[key] = (
                    memory_overhead[key] +
                    control_overhead[key] +
                    compute_overhead[key]
            )
        return total_resources