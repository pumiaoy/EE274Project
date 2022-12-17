from naive_multiwire_tensor import *

# Author: Dante Muratore


# Paths:
data_path = '/home/pumiaoy/DATA/'
#analysis_path = '/home/guest/dantemur/analysis/'
#dataset = '2015-11-09-3/'
#dataset = '2016-02-17-5/'
dataset = '2019-11-07-5/'

#dataset = '2020-01-30-5/'
#inp_file = data_path + dataset + 'orig/mat/data000_360s.mat'
inp_file = data_path + dataset + 'mat/data000_1200s.mat'
#inp_file = '/home/pumiaoy/mat/data000_1800s.mat'
strategy = 'naive_2ramp/' + dataset +'4Proj/'
subset = 'data000'
#vision_path = '/home/guest/dantemur/utilities/vision7-unix'
#vision_rgb = vision_path + '/RGB-8-1-0.48-11111.xml'


# Inputs:
time_window = 1200 # number of time samples in data in seconds
chunk_size =512 # chunk of samples to be processed simultaneously
num_bits = [8]
n_wires = [4]
ovr = True
save_path = data_path + dataset + strategy + str(num_bits) + 'b_' + str(n_wires) + 'w/' + subset

for b in range(len(num_bits)):
    for w in range(len(n_wires)):
        print('Running naive decoder with ' + str(num_bits[b]) + ' bits and ' + str(n_wires[w]) + " wires")
        save_path = data_path + dataset + strategy + str(num_bits[b]) + 'b_' + str(n_wires[w]) + 'w/' + subset
        #naive_multiwire_tensor(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)
        #naive_interleaved_tensor(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)
        #get_MSE(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)
        get_compression_rate(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)