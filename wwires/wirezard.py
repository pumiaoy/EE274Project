from os.path import dirname
from os import chdir, getcwd

import torch
import matplotlib.pyplot as plt


from wwires import CmdLineParser, GraphArchitect

def random_array_batch(system):
    num_rows = system['parameters']['num_rows']
    num_cols = system['parameters']['num_cols']
    num_batches = system['parameters']['num_batches']

    return torch.randn(num_rows, num_cols, num_batches)/5

def run_batch(method=None, system_type='simple'):

    #Load the specified yaml configuration file and build an object graph
    system = GraphArchitect(system_type, path_name='../config/system')
    
    #Get short-hand variables to simplify code
    v_ref = system['parameters']['voltage_reference']
    num_bits = system['parameters']['num_bits']

    #System Parameter Dependent Parameters
    lsb =  2*v_ref/2**num_bits
    system['parameters']['lsb'] = lsb

    for ii in range(100):
        #Create a random input
        input_tensor_batch = random_array_batch(system).cuda()

        #Calculate Input Dependent System Parameters
        v_rn = (torch.min(input_tensor_batch).item()//lsb)*lsb - lsb
        v_rp = (torch.max(input_tensor_batch).item()//lsb)*lsb + lsb
        num_rlevels = int((v_rp-v_rn)/lsb + 1)

        #Put them into the Program's Configuration Dictionairy 
        system['parameters']['v_rn'] = v_rn
        system['parameters']['v_rp'] = v_rp
        system['parameters']['num_rlevels'] = num_rlevels

        output_tensor = system.execute([input_tensor_batch])

        print(output_tensor[0].size())

if __name__ == '__main__':
    cmd = CmdLineParser()
    cmd.call_with_args(run_batch)