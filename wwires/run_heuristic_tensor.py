import torch
from gpu_memory import *

from heuristic_tensor import *
from data_interface_class import *
from remove_offset import *
from interpolate_rawdata import *

import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import pdb

def rect_heuristic_tensor(inp_file, save_path, time_window, chunk_size, num_bits, ovr, P_info, max_row_col_coll):
    A = time.time()
    
    # Options and Constants:
    print_every_batch = 1  # print running every x batches
    fs = 20000
    v_ref = 2048
    num_samples = int(time_window*fs)
    array = [16,32]
#     decoded_data_mse = np.zeros((20000, array[0]*array[1]))
#     compression_batch = np.zeros((20000, array[0]*array[1]))
#     bit_rate_batch = np.zeros((20000, array[0]*array[1]))

    # Generate an IO Object:
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)
    
    pos_log = sio.loadmat('posLog.mat')['posLog'].astype('int') - 1 #-1 since bringing matlab indices to python
    num_channels = int(array[0]*array[1])
    idx_dim0 = pos_log[:,0]
    idx_dim0 = np.reshape(idx_dim0,(num_channels,1))
    idx_dim1 = pos_log[:,1]
    idx_dim1 = np.reshape(idx_dim1,(num_channels,1))
    
    if torch.cuda.is_available():
        cuda = torch.device("cuda")  # a CUDA device object
        torch.cuda.empty_cache()
    
    mse, energy = 0, 0

    for batch_num in range(run1.num_batches):
        torch.cuda.empty_cache()
        
        # read batch and extract TTL channel
        run1.read_batch(batch_num, print_every_batch)
        samples_batch = run1.raw_data_batch.shape[1];
        processed_data_ttl = np.zeros((1,samples_batch),dtype='int16')
        processed_data_ttl = run1.raw_data_batch[0:1,:].astype('int16')
        
        # remove offset from data channels
        # use tensors from here
        processed_data = torch.zeros(samples_batch,num_channels)
        processed_data = torch.einsum('ij->ji',(torch.from_numpy(run1.raw_data_batch[1:,:]),))
        processed_data = processed_data.to(cuda)
        processed_data_mean = torch.zeros_like(processed_data)
        processed_data_mean = remove_offset(processed_data)
        
        if batch_num%print_every_batch == 0:
            print('Preprocessing Done')
            
        # move data from [num_channels,n_samples] to [x_array,y_array,n_samples]
        array_in = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
        array_in[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(processed_data_mean,)).view(array[0]*array[1],1,samples_batch)
        
        # free GPU space
        del processed_data
        torch.cuda.empty_cache()
        
        ###### Decode Rect Heuristic #######
#         decoded_data = heuristic_tensor(array, array_in, num_bits, v_ref,
#                                                             samples_batch, idx_dim0, idx_dim1, cuda, P_info)
        decoded_data, decoded_data_naive, compression, bit_rate = heuristic_tensor(array, array_in, num_bits, v_ref,
                                                            samples_batch, idx_dim0, idx_dim1, cuda, P_info, 
                                                            collision = max_row_col_coll)
        
#         decoded_data_mse[batch_num*samples_batch:(batch_num+1)*samples_batch,:] = decoded_data
        
#         compression_batch[batch_num*samples_batch:(batch_num+1)*samples_batch,:] = compression
#         bit_rate_batch[batch_num*samples_batch:(batch_num+1)*samples_batch,:] = bit_rate

        ####################################
        del array_in
        torch.cuda.empty_cache()
        
         # interpolation is done on the accumulated value
        naive_interp = interpolate_rawdata(decoded_data, samples_batch, cuda, sigma=5)
#         naive_interp = interpolate_rawdata_nonoise(decoded_data, samples_batch, cuda)
        
        del decoded_data, decoded_data_naive
        torch.cuda.empty_cache()
# ##############################################################################################################        
######### Saving Data:
        #diff_plot(processed_data_mean.to('cpu').numpy(), naive_interp.to('cpu').numpy(), 500)

        # save a few unaltered data samples for correct noise/threshold calcuation in Vision at the beginning
        if batch_num < 100: #2: #10s same data as original
            decoded_out = torch.einsum('ij->ji',(processed_data_mean,)).to('cpu').numpy().astype('int16')
        else:
            decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
        
#         del processed_data_mean, naive_interp
        torch.cuda.empty_cache()
        
        # save decoded data to .bin files
        run1.output_data_batch = np.concatenate((processed_data_ttl, decoded_out)) # decoded_data should be 512*chunk_size
        run1.generate_header()
        run1.generate_compressed_data()
        run1.write_bin_file()
        
        B = time.time()
        
#         print('Time for batch of size ' + str(samples_batch) + ' = ' + str(B-A))
        if batch_num%print_every_batch == 0:
            print('Time = ' + str(B-A))
    
       
#         print('Absolute Difference (L1 dist) between heuristic and naive data = ' + str(
#             torch.sum(torch.abs(decoded_data - decoded_data_naive)).item()))
      
##############################################################################################################        
##### Visualization:
# # #         pdb.set_trace()
#         n_ch = np.concatenate((np.array([0,1]),np.random.randint(512, size=2))) #[1] #np.random.randint(512, size=5)
#         for ch in n_ch:
#             plt.figure()
#             plt.plot(np.arange(samples_batch),processed_data_mean[:,ch].to('cpu').numpy())
#             plt.plot(np.arange(samples_batch),decoded_data_naive[:,ch].to('cpu').numpy(),color='g', marker='.')
#             plt.plot(np.arange(samples_batch),decoded_data[:,ch].to('cpu').numpy(),color='r', linestyle=':', marker='o')
#             plt.plot(np.arange(samples_batch),naive_interp[:,ch].to('cpu').numpy(),color='m', linestyle=':', marker = '*')       
#             plt.title('Channel = ' + str(ch))
#             plt.legend(['Raw','Naive','Heuristic','Interpolated Heuristic'])
#             plt.figure()
#             plt.plot(np.arange(samples_batch),decoded_data[:,ch].to('cpu').numpy()-decoded_data_naive[:,ch].to('cpu').numpy())
#             plt.plot(np.arange(samples_batch),(decoded_data_naive[:,ch].to('cpu').numpy())!=0,color='g', marker='o')
#             plt.title('Diff: Heuristic - Naive')
#             plt.legend(['Diff', 'Naive Unique Points'])
#         plt.show()
        
        
#         pdb.set_trace()
#         save_dict = {}
#         save_dict['decoded_data_naive_'+str(num_bits)] = decoded_data_naive.to('cpu').numpy()
#         save_dict['decoded_data_heuristic_'+str(num_bits)] = decoded_data.to('cpu').numpy()
#         save_dict['interpolated_data_heuristic_'+str(num_bits)] = naive_interp.to('cpu').numpy()

#         sio.savemat('../../results/b_' + str(num_bits) + '_coll_' + str(max_row_col_coll), save_dict)
#         print('Mat file saved')

##############################################################################################################        
#### MSE and Compression:
#         mse += np.sum((naive_interp.to('cpu').numpy() - processed_data_mean.to('cpu').numpy())**2) #np.mean((naive_interp.to('cpu').numpy() - processed_data_mean.to('cpu').numpy())**2) #np.sum((naive_interp.to('cpu').numpy() - processed_data_mean.to('cpu').numpy())**2) 
#         energy += np.sum(processed_data_mean.to('cpu').numpy()**2) #np.var(processed_data_mean.to('cpu').numpy())
    
# #     mse = mse/run1.num_batches
#     energy_hc = 72.6056*512*20000
#     nmse = mse/energy#_hc
#     nmse_db = 10*np.log(nmse)/np.log(10)
#     print('Collision cutoff = ' + str(max_row_col_coll))
#     print('r_firing = ' + str(P_info['r_firing']))
#     print('Energy = ' + str(energy))
#     print('MSE = ' + str(mse))
#     print('NMSE = ' + str(nmse))
#     print('NMSE (in db) = ' + str(nmse_db))

#     sio.savemat('../../results/b_' + str(num_bits) + '_mse', {'decoded_data' : decoded_data_mse})
#     print('Saved data to calculate MSE')
    
#     print('Bit Rate for ' + str(num_bits) + ' bits heuristic is = ' + str(np.mean(bit_rate_batch)))
#     print('Compression for ' + str(num_bits) + ' bits heuristic is = ' + str(np.mean(compression_batch)))
