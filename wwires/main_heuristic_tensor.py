from run_heuristic_tensor import *

# Paths:
data_path = '/home/guest/tpulkit/data/'
analysis_path = '/home/guest/tpulkit/analysis/'
dataset = '2015-11-09-3/'
inp_file = data_path + dataset + 'orig/mat/data000_360s_copy.mat'
strategy = 'heuristic/'
subset = 'data000'
vision_path = '/home/guest/dantemur/utilities/vision7-unix'
vision_rgb = vision_path + '/RGB-8-1-0.48-11111.xml'

# Inputs:
time_window = 0.1*0.2 #60*6 #0.1*0.2 #1 #0.1*0.2 #6*60 #0.5 #0.1*0.75 #60*6 # number of time samples in data in seconds
chunk_size = 2000 #400 #1500 # chunk of samples to be processed simultaneously
num_bits = [10] #[10] #[12] #[8,9,11,12]
ovr = False 
pi, pd, rf = 1.5, 0.9, 1 #1.5, 0.9, 2 #2, 0.99, 2 #1.5, 0.9, 5 #1.5, 0.9, 1 #1.5, 0.9, 5 #1.5, 0.9, 10 #1.5, 0.9, 1 
        
P_info = {'P_inc': pi,
          'P_dec': pd,
          'r_firing': rf}

max_row_col_coll = 2

#save_path = data_path + dataset + strategy + str(num_bits) + 'b_' + str(n_wires) + 'w/' + subset

for b in range(len(num_bits)):
    print('Running naive decoder with ' + str(num_bits[b]) + ' bits')
    save_path = data_path + dataset + strategy + \
                str(num_bits[b]) + 'b_' + str(rf) + 'r_' + \
                str(max_row_col_coll) + 'c_noise/' + subset
    rect_heuristic_tensor(inp_file, save_path, time_window, chunk_size, num_bits[b], ovr, P_info, max_row_col_coll)
        


