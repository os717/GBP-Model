from Hardware_Model.library.model import Hardware_Model

import argparse
import math
import json
from itertools import product
from datetime import datetime

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate Designs')
parser.add_argument('-save', type=str, help='name of file to be saved', default="")
filename = parser.parse_args().save

""" UTIL  FUNCTIONS """
def design_resources_less_than_available(board, design_resources, fpga_boards, safety_factor):
    if board in fpga_boards:
        board_resources = fpga_boards[board]
        for key, value in design_resources.items():
            if key in board_resources and value <= board_resources[key]*safety_factor:
                continue
            else:
                return False
        return True
    else:
        print("Board not found in FPGA boards dictionary.")
        return False

""" SETUP """

# FPGA resources
fpga_boards = {
    'pynq_z1': {
        'lut': 53200,
        'ff': 106400,
        'bram': 140,
        'dsp': 220
    },
    'Ultrascale': {
        'lut': 1182240,
        'ff': 2364480,
        'bram': 2160,
        'dsp': 6840
    }
}

# graph topology
graph_size = 1000
num_factors = 1000

# fpga specifications
fpga_resources = fpga_boards['pynq_z1']
fpga_clk = 1/(100*math.pow(10, 6))
memory_bw = 1/(100*math.pow(10, 6))

""" GENERATE DESIGNS """

# node updates per PE
inp_nodes_updt_list = [i for i in range(1, graph_size+1)]

# compute unit parameterisations
compute_blocks = ['inv', 'mul', 'add', 'sub']
unroll_factor_lists = list(product([1,3], repeat=4))
inp_compute_unit_parameterisations = [{compute_blocks[i]: sublist[i] for i in range(len(compute_blocks))} for sublist in unroll_factor_lists] 

# double buffering parameters
inp_double_buffering = [True, False]

# safety factor
inp_safety_factor = 0.65

# fpga board
inp_fpga_board = 'pynq_z1'

# capping
inp_capping = [None, 1]

# node update policy
node_update_policy = ['random', 'fixed']

# store all designs
DESIGNS = []

# iterate over designs

total_iterations = len(inp_nodes_updt_list) * len(inp_compute_unit_parameterisations) * len(inp_double_buffering) * len(inp_capping) * len(node_update_policy)
pbar = tqdm(total=total_iterations, desc="Generating Designs", unit="designs")

for policy in node_update_policy:
    for node in inp_nodes_updt_list:
        for compute_params in inp_compute_unit_parameterisations:
            for double_buffering in inp_double_buffering:
                for capping in inp_capping:
                    # iterate over number of pes
                    number_pes = 1
                    while True:
                        # generate design
                        design = Hardware_Model.generate(graph_size=graph_size, 
                                        num_factors=num_factors, 
                                        fpga_resources=fpga_resources, 
                                        fpga_clk=fpga_clk, 
                                        memory_bw=memory_bw,
                                        nodes_updt_per_pe=node, 
                                        number_pes=number_pes, 
                                        compute_unroll_factor=compute_params,
                                        double_buffering=double_buffering,
                                        capping=capping)
                        design['design']['policy'] = policy
                        # compare resources to safety factor
                        if not design_resources_less_than_available(inp_fpga_board, design['resources']['resources_total'], fpga_boards, inp_safety_factor):
                            break
                        if design['design']['nodes_updt_per_stream'] > graph_size:
                            break
                        else:
                            number_pes += 1
                            DESIGNS.append(design)
                    pbar.update(1)
pbar.close()

""" STORE DESIGNS """

# Combine predefined values with designs
data = {
    'inp_nodes_updt_list': inp_nodes_updt_list,
    'compute_blocks': compute_blocks,
    'inp_compute_unit_parameterisations': inp_compute_unit_parameterisations,
    'inp_double_buffering': inp_double_buffering,
    'inp_safety_factor': inp_safety_factor,
    'inp_fpga_board': inp_fpga_board,
    'inp_graph_topology' : {'N': graph_size, 'k': num_factors},
    'all_designs': DESIGNS
}

filename_to_save_design = f"Hardware_Model/designs/designs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json" if filename == "" else f"Hardware_Model/designs/{str(filename)}.json"

# Write the list of dictionaries to the file
with open(filename_to_save_design, 'w') as file:
    json.dump(data, file, indent=4)