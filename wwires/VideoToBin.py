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
import imageio

data_path = '/home/pumiaoy/DATA/'
dataset = '2015-11-09-7/'
inp_file = data_path + 'data000_1800s.mat'
#inp_file = '/home/pumiaoy/mat/data000_1800s.mat'
strategy = 'naive_2ramp/' + dataset +'8bit/'
subset = 'data000'
save_path = '/home/pumiaoy/stanford_compression_library/Project/Data/8bit_tests'



def remove_offset(raw_data):
    minumum, indexs= torch.min(raw_data, 1, True)
    raw_data_pp_mean = raw_data - torch.mean(raw_data, 1, True)
    
    return raw_data_pp_mean.type(torch.int16)






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
time_window = 60  #360 # number of time samples in data in seconds
chunk_size = 1024 # chunk of samples to be processed simultaneously
num_bits = 11
n_wires = 16

#f = h5py.File('/home/pumiaoy/stanford_compression_library/Project/Data/data000_1200s.mat')
#APData=f['newData']   

t_clip =100
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

# read video into 3d array

frames = []


path = "/home/pumiaoy/stanford_compression_library/Project/compressed_265_28.mp4" 
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray_img)
video_comp = np.stack(frames, axis=0) # dimensions (T, H, W, C)

frames_raw=[]
path_1 = "/home/pumiaoy/stanford_compression_library/Project/Raw_video_1m.mp4" 
cap_1 = cv2.VideoCapture(path)
ret = True
while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames_raw.append(gray_img)
video_raw = np.stack(frames_raw, axis=0) # dimensions (T, H, W, C)


fig = plt.figure(dpi=300)

print(np.shape(video))

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
number_batch= run1.num_batches
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

    

    # interpolation is done on the accumulated value
    naive_interp = interpolate_rawdata(naive_decoded, samples_batch, cuda, sigma=0)

    # save a few unaltered data samples for correct noise/threshold calcuation in Vision at the beginning
    # if batch_num < 2:
    #     decoded_out = torch.einsum('ij->ji',(processed_data_mean,)).to('cpu').numpy().astype('int16')
    # else:
    #     decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
    #     #diff_plot(processed_data_mean.to('cpu').numpy(), np.einsum('ij->ji',decoded_out), 1, batch_num)

    decoded_out = torch.einsum('ij->ji',(naive_interp,)).to('cpu').numpy().astype('int16')
    #print(np.shape(decoded_out))
    HFS = torch.ones(n_channels,samples_batch,dtype = torch.int16) * (2**(7))

    decoded_out_tensor = torch.from_numpy(decoded_out)
    decoded_out_tensor = decoded_out_tensor+HFS
    decoded_out_tensor = torch.einsum('ij->ji',(decoded_out_tensor,))
    #print(np.shape(decoded_out_tensor))
    array_output = torch.zeros((array[0],array[1],samples_batch),dtype=torch.int16, device = cuda)
    array_output[idx_dim0,idx_dim1,:] = torch.einsum('ij->ji',(decoded_out_tensor,)).view(array[0]*array[1],1,samples_batch).to(cuda)
    #print(array_output)
    #plt.figure()
    #plt.plot(decoded_out[100,:])
    #plt.show()
    #print(np.shape(array_output))
    #print(torch.min(array_output))
    #print(torch.max(array_output)-torch.min(array_output))
    # remove the first two batches
    # Make all positive +128
    array_output =  array_output.to('cpu')
    array_output = array_output.numpy()
    batch_data = np.uint8(array_output)
    #print(np.shape(array_output))
    if batch_num == 0:
        data = batch_data
    else:
        data = np.concatenate((data, batch_data),axis = 2)
    #Make video example
    # size = 16, 32
    # duration = 32
    # fps = 32
    # fourcc = cv2.VideoWriter_fourcc('v','p','0','9')
    # #fourcc = -1#cv2.VideoWriter_fourcc(*'H264')
    # out = cv2.VideoWriter('/home/pumiaoy/stanford_compression_library/Project/001.mp4', fourcc, fps, (size[1], size[0]), False)
    # for frame in range(fps * duration):
    #     vdata = data[:,:,frame]
    #     out.write(vdata)
    # out.release()



    #Save as video 
    #Compression 


    # save decoded data to .bin files
    run1.output_data_batch = np.concatenate((processed_data_ttl, decoded_out)) # decoded_data should be 512*chunk_size
    run1.generate_header()
    run1.generate_compressed_data()
    run1.write_bin_file()

B = time.time()
print('Time taken for decoding ' + str(num_samples) + ' ramp samples = ' + str(B-A))

# size = 16, 32
# duration = 32*number_batch-10
# fps = 32

# import skvideo.io

# #imageio.mimwrite('/home/pumiaoy/stanford_compression_library/Project/Raw_video.mp4', data , fps = float(20))
# writer = skvideo.io.FFmpegWriter("/home/pumiaoy/stanford_compression_library/Project/Raw_video.mp4")
# for frame in range(fps * duration):
#         writer.writeFrame(data[:,:,frame])
# writer.close()
# # fourcc = cv2.VideoWriter_fourcc('v','p','0','9')
# # #fourcc = -1#cv2.VideoWriter_fourcc(*'H264')
# # out = cv2.VideoWriter('/home/pumiaoy/stanford_compression_library/Project/full.mp4', fourcc, fps, (size[1], size[0]), False)
# # for frame in range(fps * duration-2):
# #     vdata = data[:,:,frame]
# #     out.write(vdata)
# # out.release()