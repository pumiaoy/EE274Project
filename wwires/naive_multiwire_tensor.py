from data_interface_class import *
from remove_offset import *
from naive_tensor import *
from interpolate_rawdata import *
from simple_plot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import numpy as np
import time
import torch

def naive_multiwire_tensor(inp_file, save_path, time_window, chunk_size, num_bits, n_wires, ovr):
    A = time.time()

    # Options and Constants:
    print_every_batch = 50  # print running every x batches
    fs = 20000
    v_ref = 2048# while the max value is 387 in this dataset
    num_samples = int(time_window*fs)
    array = [16,32]

    # Generate an IO Object:
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)

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
        processed_data_mean = remove_offset(processed_data)

        if batch_num%print_every_batch == 0:
            print('Preprocessing Done')

        # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)

        # free GPU space
        del processed_data
        torch.cuda.empty_cache()

        # encode and decode data with naive decoder
        array_in = array_in.to(cuda)   # send input data to GPU
        naive_decoded = torch.zeros(samples_batch,n_channels, device = cuda)
        #naive_decoded_v2 = torch.zeros(samples_batch,n_channels, device = cuda)
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
            #naive_decoded    = naive_decoded + rect_naive_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
            naive_decoded    = naive_decoded + rect_naive_4proj_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
            #naive_decoded_v2 = naive_decoded_v2 + rect_naive_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)


            mask = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
            #diff_plot(processed_data_mean.to('cpu').numpy(), naive_decoded_v2.to('cpu').numpy(), 1, batch_num)
            #diff_plot(naive_decoded.to('cpu').numpy(), naive_decoded_v2.to('cpu').numpy(), 1, samples_batch)

        # interpolation is done on the accumulated value
        #naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=5.2)
        naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=0)
        #naive_interp_v2 = interpolate_rawdata(naive_decoded_v2, samples_batch, cuda, sigma=2)
        #naive_interp = processed_data_mean
        temp=  np.zeros((chunk_size,512))
        x = np.arange(512)+temp
        #plt.scatter(x,naive_decoded.cpu().numpy())
        #plt.plot(naive_decoded.cpu().numpy())

        #plt.plot(naive_interp_v2.cpu().numpy()[:,2])

        plt.show()

        # save a few unaltered data samples for correct noise/threshold calcuation in Vision at the beginning
        if batch_num < 2:
            decoded_out = torch.einsum('ij->ji',(processed_data_mean,)).to('cpu').numpy().astype('int16')
        else:
            decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
            #diff_plot(processed_data_mean.to('cpu').numpy(), np.einsum('ij->ji',decoded_out), 1, batch_num)

        # save decoded data to .bin files
        run1.output_data_batch = np.concatenate((processed_data_ttl, decoded_out)) # decoded_data should be 512*chunk_size
        run1.generate_header()
        run1.generate_compressed_data()
        run1.write_bin_file()

    B = time.time()
    print('Time taken for decoding ' + str(num_samples) + ' ramp samples = ' + str(B-A))
    return run1.output_data_batch


def naive_interleaved_tensor(inp_file, save_path, time_window, chunk_size, num_bits, n_wires, ovr):
    A = time.time()

    # Options and Constants:
    print_every_batch = 50  # print running every x batches
    fs = 20000
    v_ref = 2048
    num_samples = int(time_window*fs)
    array = [16,32]

    # Generate an IO Object:
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)

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
        processed_data_mean = remove_offset(processed_data)

        if batch_num%print_every_batch == 0:
            print('Preprocessing Done')

        # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)

        # free GPU space
        del processed_data
        torch.cuda.empty_cache()

        # encode and decode data with naive decoder
        array_in = array_in.to(cuda)   # send input data to GPU
        naive_decoded = torch.zeros(samples_batch,n_channels, device = cuda)
        naive_decoded = rect_naive_interleaved_tensor\
                (array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)

        # interpolation is done on the accumulated value
        naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=0)
        #naive_interp = naive_decoded


        # save a few unaltered data samples for correct noise/threshold calcuation in Vision at the beginning
        if batch_num < 2:
            decoded_out = torch.einsum('ij->ji',(processed_data_mean,)).to('cpu').numpy().astype('int16')
        else:
            decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
            #diff_plot(processed_data_mean.to('cpu').numpy(), np.einsum('ij->ji', decoded_out), 416, batch_num)


        # save decoded data to .bin files
        run1.output_data_batch = np.concatenate((processed_data_ttl, decoded_out)) # decoded_data should be 512*chunk_size
        run1.generate_header()
        run1.generate_compressed_data()
        run1.write_bin_file()

    B = time.time()
    print('Time taken for decoding ' + str(num_samples) + ' ramp samples = ' + str(B-A))

def get_compression_rate(inp_file, save_path, time_window, chunk_size, num_bits, n_wires, ovr):
    A = time.time()

    # Options and Constants:
    print_every_batch = 100000  # print running every x batches
    fs = 20000
    v_ref = 2048
    num_samples = int(time_window*fs)
    array = [16,32]

    # Generate an IO Object:
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)

    # Import poslin for getting elmap:
    pos_lin = sio.loadmat('posLin.mat')['posLin'].astype('int') - 1 #-1 since bringing matlab indices to python
    pos_log = sio.loadmat('posLog.mat')['posLog'].astype('int') - 1 #-1 since bringing matlab indices to python
    n_channels = pos_lin.shape[0]
    #idx_dim0 = pos_lin // array[1]
    #idx_dim1 = pos_lin % array[1]
    idx_dim0 = pos_log[:,0]
    idx_dim0 = np.reshape(idx_dim0,(n_channels,1))
    idx_dim1 = pos_log[:,1]
    idx_dim1 = np.reshape(idx_dim1,(n_channels,1))

    if torch.cuda.is_available():
        cuda = torch.device("cuda")  # a CUDA device object

    bit_tx = 0
    for batch_num in range(run1.num_batches):
        # read batch and extract TTL channel
        #print(batch_num)
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
        processed_data_mean = remove_offset(processed_data)

        # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)

        # free GPU space
        del processed_data
        torch.cuda.empty_cache()

        # encode and decode data with naive decoder
        array_in = array_in.to(cuda)   # send input data to GPU
        bit_tx_batch = 0
        mask = torch.zeros(array[0],array[1],1, device=cuda, dtype=torch.int16)
        # each wire configuration presents a new array to the encoder/decoder
        # outputs are accumulated
        for current_wire in range(n_wires):
            lin_mask = torch.arange(current_wire,n_channels,n_wires)
            #x_mask = lin_mask[current_wire:n_channels//n_wires]//array[1]
            #y_mask = lin_mask[current_wire:n_channels//n_wires]%array[1]
            x_mask = lin_mask % array[0]
            y_mask = lin_mask // array[0]
            mask[x_mask,y_mask] = 1
            array_to_decoder = array_in*mask
            bit_tx_batch = bit_tx_batch + (np.log2(array[1]) + np.log2(array[0]/n_wires))*rect_naive_tensor_profile \
                (array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
            mask = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)

        bit_tx = bit_tx + bit_tx_batch
    bit_tx = bit_tx/run1.num_batches

    B = time.time()
    print('Average bits transmitted per sample: ' + str(bit_tx.to('cpu').numpy()) + ' bits')
    print('Average valid channels per sample: '+
          str(bit_tx.to('cpu').numpy()/(np.log2(array[1]) + np.log2(array[0]/n_wires))) + ' channels\n\n')

def get_MSE(inp_file, save_path, time_window, chunk_size, num_bits, n_wires, ovr):

    # Options and Constants:
    print_every_batch = 50  # print running every x batches
    fs = 20000
    v_ref = 2048
    num_samples = int(time_window*fs)
    array = [16,32]

    # Generate an IO Object:
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)

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

    MSE = 0

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
        processed_data_mean = remove_offset(processed_data)

        if batch_num%print_every_batch == 0:
            print('Preprocessing Done')

        # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)

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
            naive_decoded = naive_decoded + rect_naive_3proj_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
            mask = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
            #diff_plot(processed_data_mean.to('cpu').numpy(), naive_decoded.to('cpu').numpy(), 1, batch_num)

        # interpolation is done on the accumulated value
        naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=2)

        MSE = MSE + torch.mean(torch.mean((processed_data_mean.float() - naive_interp)**2,0))

    MSE = MSE/run1.num_batches
    return MSE

def get_NMSE(inp_file, save_path, time_window, chunk_size, num_bits, n_wires, ovr):

    # Options and Constants:
    print_every_batch = 50  # print running every x batches
    fs = 20000
    v_ref = 2048
    num_samples = int(time_window*fs)
    array = [16,32]

    # Generate an IO Object:
    run0 = io_object(time_window, time_window*fs, inp_file = inp_file, save_path = save_path, overwrite = ovr)
    run0.read_batch(0, print_every_batch)
    inputPower = run0.raw_data_batch.var()

    run1 = io_object(time_window, chunk_size, inp_file=inp_file, save_path=save_path, overwrite=ovr)

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

    MSE = 0

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
        processed_data_mean = remove_offset(processed_data)

        if batch_num%print_every_batch == 0:
            print('Preprocessing Done')

        # move data from [n_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)

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
            naive_decoded = naive_decoded + rect_naive_tensor(array, array_to_decoder, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda)
            mask = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
            #diff_plot(processed_data_mean.to('cpu').numpy(), naive_decoded.to('cpu').numpy(), 1, batch_num)

        # interpolation is done on the accumulated value
        naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=2)

        MSE = MSE + torch.mean(torch.mean((processed_data_mean.float() - naive_interp)**2,0))

    MSE = MSE/run1.num_batches

    NMSE = MSE/inputPower
    return NMSE
