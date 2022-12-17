## This script requantizes the raw recording
# read mat file
# change array shape
# requantize to 8bit
# make video
# video compression
# convert back to 512 shape
# save as bin files 
from data_interface_class import *
from simple_plot import *
import scipy.io as sio
import numpy as np
import time
import torch
import ffmpeg
import h5py
import torch
import cv2
import matplotlib.pyplot as plt

data_path = '/home/pumiaoy/DATA/'
dataset = '2015-11-09-7/'
inp_file = data_path + 'data000_1800s.mat'
#inp_file = '/home/pumiaoy/mat/data000_1800s.mat'
strategy = 'naive_2ramp/' + dataset +'8bit/'
subset = 'data000'
save_path = '/home/pumiaoy/stanford_compression_library/Project/Data/8bit_1200s'





# todo
# requantize value
# have outputs be unsigned int8 --> positive numbers
# convert to video

def remove_offset(raw_data):
    minumum, indexs= torch.min(raw_data, 1, True)
    raw_data_pp_mean = raw_data - torch.mean(raw_data, 1, True)
    
    return raw_data_pp_mean.type(torch.int16)



def quantize_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    # generate rampup and rampdown for comparators and add dimentions for comparison
    n_channels = array[0]*array[1]
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    v_r = max(abs(v_rn),abs(v_rp))
    rampup = torch.linspace(-v_r, v_r, int( 2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampup = rampup.view(rampup.shape[0],1,1,1)
    rampup_next = torch.linspace(-v_r+lsb, v_r+lsb, int(2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampup_next = rampup_next.view(rampup_next.shape[0],1,1,1)
    rampdown = torch.linspace(v_r, -v_r, int( 2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampdown = rampdown.view(rampdown.shape[0],1,1,1)
    rampdown_next = torch.linspace(v_r+lsb, -v_r+lsb, int(2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampdown_next = rampdown_next.view(rampdown_next.shape[0],1,1,1)
    # create mask for up/down, create two comp tensors, combine them for validation but decode them separately
    maskup = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
    lin_maskup = torch.arange(0, n_channels, 2)
    x_maskup = lin_maskup % array[0]
    y_maskup = lin_maskup // array[0]
    maskup[x_maskup, y_maskup] = 1
    maskdown = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
    lin_maskdown = torch.arange(1, n_channels, 2)
    x_maskdown = lin_maskdown % array[0]
    y_maskdown = lin_maskdown // array[0]
    maskdown[x_maskdown, y_maskdown] = 1
    # create tensor of comparator outputs
    compup = torch.ge(array_in*maskup,rampup) & torch.lt(array_in*maskup,rampup_next)
    compdown = torch.ge(array_in * maskdown, rampdown) & torch.lt(array_in * maskdown, rampdown_next)
    # extract row and column tensors for all samples and ramp values, both for up and down
    rowup = compup.any(dim=2)
    colup = compup.any(dim=1)
    rowdown = compdown.any(dim=2)
    coldown = compdown.any(dim=1)
    # combine row and column to get valid tensor
    row = rowup|rowdown
    col = colup|coldown
    # free some space
    del compup, compdown, array_in, rowup, rowdown,
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = torch.einsum("ij->ji", (valid,))
    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0],row.shape[1],row.shape[2],1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    # free some space
    del row, col
    torch.cuda.empty_cache()
    # go back to all channels in one vector
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,rampup.shape[0],array[0]*array[1])
    maskup_out = maskup[idx_dim0, idx_dim1,0].view(1,1,n_channels)
    maskdown_out = maskdown[idx_dim0, idx_dim1,0].view(1,1,n_channels)
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)
    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out,valid.float())
    # free some space
    del naive_out
    torch.cuda.empty_cache()
    # reshape ramp for torch.einsum
    #rampup = rampup.view(rampup.shape[0])
    rampup_ADC =torch.linspace(int(-v_r/lsb), int(v_r/lsb), int( 2*v_r/lsb + 1), device = cuda).type(torch.int16)
    #rampup_ADC =rampup_ADC.view(rampup_ADC.shape[0],1,1,1)#Reshape
    #rampdown = rampdown.view(rampup.shape[0])
    rampdown_ADC = torch.linspace(int(v_r/lsb), int(-v_r/lsb), int( 2*v_r/lsb + 1), device = cuda).type(torch.int16)
    #rampdown_ADC = rampdown_ADC.view(rampup_ADC.shape[0])
    #maskup = torch.zeros(1,1,array[0]*array[1], device=cuda, dtype=torch.int16)
    #maskup[0,0,lin_maskup] = 1
    #maskdown = torch.zeros(1, 1, array[0] * array[1], device=cuda, dtype=torch.int16)
    #maskdown[0,0,lin_maskdown] = 1
    # use torch.einsum to sum across all ramp values and get decoded output
    a = naive_valid*maskup_out.float()
    #print(rampup.size())
    #print(a.size())
    #print(rampup_ADC.size())
    #print(rampup_ADC)
    naive_decoded = torch.einsum("ijk,j->ik", (naive_valid*maskup_out.float(),rampup_ADC.float())) + \
                    torch.einsum("ijk,j->ik", (naive_valid*maskdown_out.float(),rampdown_ADC.float()))
    return naive_decoded


def interpolate_rawdata(decoded_data,samples_batch,cuda,sigma):
    #decoded_data = torch.tensor([[1, 0, 0, 2, 4, 0, 1],[1, 0, 0, 2, 4, 0, 1],[1, 0, 0, 2, 4, 0, 1]]).type(torch.FloatTensor)
    #decoded_data = torch.einsum('ij->ji', (decoded_data,)).to(cuda)
    #samples_batch= decoded_data.shape[0]
    interpolated_data = torch.zeros((samples_batch, decoded_data.shape[1]), device = cuda)
    
    logic_matrix_zero = torch.eq(decoded_data[1:samples_batch-1,:],0).type(torch.FloatTensor).to(cuda)
    perm_previous = torch.arange(0,samples_batch-2).type(torch.LongTensor).to(cuda)
    perm_next = torch.arange(2,samples_batch).type(torch.LongTensor).to(cuda)
    interpolated_data[1:samples_batch-1,:] = decoded_data[1:samples_batch-1,:]
    return interpolated_data





# Inputs:
time_window = 1 # number of time samples in data in seconds
chunk_size = 500 # chunk of samples to be processed simultaneously
num_bits = 8
n_wires = 16

#f = h5py.File('/home/pumiaoy/stanford_compression_library/Project/Data/data000_1200s.mat')
#APData=f['newData']   

t_clip = 1
fs = 20000
nS =  2000
array =  [16,32,nS,3]
n_channels = 512
#APclip = APData[1:513, 0:nS]
A = time.time()
print_every_batch = 50  # print running every x batches
fs = 20000
v_ref = 2048
num_samples = int(time_window*fs)
array = [16,32]

# Generate an IO Object:
run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path)

# Import poslin for getting elmap:
pos_lin = sio.loadmat('posLin.mat')['posLin'].astype('int') - 1 #-1 since bringing matlab indices to python
pos_log = sio.loadmat('posLog.mat')['posLog'].astype('int') - 1 #-1 since bringing matlab indices to python
n_channels = pos_lin.shape[0]
idx_dim0 = pos_log[:,0]
idx_dim0 = np.reshape(idx_dim0,(n_channels,1))
idx_dim1 = pos_log[:,1]
idx_dim1 = np.reshape(idx_dim1,(n_channels,1))

if torch.cuda.is_available():
    cuda = torch.device("cuda")  # a CUDA device object
torch.cuda.set_device(1)

for batch_num in range(run1.num_batches):
    # read batch and extract TTL channel
    run1.read_batch(batch_num, print_every_batch)
    samples_batch = run1.raw_data_batch.shape[1];
    processed_data_ttl = np.zeros((1,samples_batch),dtype='int16')
    processed_data_ttl = run1.raw_data_batch[0:1,:].astype('int16')

    # remove offset from data channels
    # use tensors from here
    processed_data = torch.zeros(samples_batch,n_channels)
    processed_data = torch.einsum('ij->ji',(torch.from_numpy(run1.raw_data_batch[1:,:]),))
    processed_data = processed_data.to(cuda)
    processed_data_mean = torch.zeros_like(processed_data)
    #
    minumum, indexs= torch.min(processed_data, 1, True)
    #
    #!!!!
    processed_data_mean = remove_offset(processed_data)
    #print(np.shape(processed_data_mean))
    if batch_num%print_every_batch == 0:
        print('Preprocessing Done')

    # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
    array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
    array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)
    #print(np.shape(array_in))
    #print(array_in)
    # free GPU space
    del processed_data
    torch.cuda.empty_cache()

    # encode and decode data with naive decoder
    array_in = array_in.to(cuda)   # send input data to GPU
    naive_decoded = torch.zeros(samples_batch,n_channels, device = cuda)
    mask = torch.zeros(array[0],array[1],1, device=cuda, dtype=torch.int16)
    # each wire configuration presents a new array to the encoder/decoder
    # outputs are accumulated
    #naive_interp = rect_naive_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)

    for current_wire in range(n_wires):
        lin_mask = torch.arange(current_wire,n_channels,n_wires)
        #x_mask = lin_mask[current_wire:n_channels//n_wires]//array[1]
        #y_mask = lin_mask[current_wire:n_channels//n_wires]%array[1]
        x_mask = lin_mask % array[0]
        y_mask = lin_mask // array[0]
        mask[x_mask,y_mask] = 1
        array_to_decoder = array_in*mask
        naive_decoded = naive_decoded + quantize_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
        mask = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
        #diff_plot(processed_data_mean.to('cpu').numpy(), naive_decoded.to('cpu').numpy(), 1, batch_num)

    # interpolation is done on the accumulated value
    naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=0)

    # save a few unaltered data samples for correct noise/threshold calcuation in Vision at the beginning
    if batch_num < 2:
        decoded_out = torch.einsum('ij->ji',(processed_data_mean,)).to('cpu').numpy().astype('int16')
    else:
        decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
        #diff_plot(processed_data_mean.to('cpu').numpy(), np.einsum('ij->ji',decoded_out), 1, batch_num)


    decoded_out_tensor = torch.from_numpy(decoded_out)
    decoded_out_tensor = torch.einsum('ij->ji',(decoded_out_tensor,))
    #print(np.shape(decoded_out_tensor))
    array_output = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
    array_output[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(decoded_out_tensor,)).view(array[0]*array[1],1,samples_batch).to(cuda)
    #print(array_output)
    #print(np.shape(array_output))
    print(torch.max(array_output)-torch.min(array_output))

    # save decoded data to .bin files
    run1.output_data_batch = np.concatenate((processed_data_ttl, decoded_out)) # decoded_data should be 512*chunk_size
    run1.generate_header()
    run1.generate_compressed_data()
    run1.write_bin_file()

B = time.time()
print('Time taken for decoding ' + str(num_samples) + ' ramp samples = ' + str(B-A))


