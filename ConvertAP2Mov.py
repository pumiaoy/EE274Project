import ffmpeg
import numpy as np
import scipy.io as sio
import h5py
import torch
import cv2
import matplotlib.pyplot as plt
f = h5py.File('./Data/data000_1200s.mat')
APData=f['newData']   

t_clip = 1
fs = 20000
nS =  100
array =  [16,32,nS,3]
n_channels = 512
APclip = APData[1:513, 0:nS]


pos_lin = sio.loadmat('./Data/posLin.mat')['posLin'].astype('int') - 1 #-1 since bringing matlab indices to python
pos_log = sio.loadmat('./Data/posLog.mat')['posLog'].astype('int') - 1 #-1 since bringing matlab indices to python
n_channels = pos_lin.shape[0]
idx_dim0 = pos_log[:,0]
idx_dim0 = np.reshape(idx_dim0,(n_channels,1))
idx_dim1 = pos_log[:,1]
idx_dim1 = np.reshape(idx_dim1,(n_channels,1))

# Reshape array from 512*1 to 16*32
# Requantize to num_bits
APclip= torch.from_numpy(APclip)
processed_data = torch.ones((1,nS))
processed_data_mean ,ind = torch.min(APclip, dim=1)
#print(processed_data_mean.size())
processed_data_mean = torch.reshape(processed_data_mean,(n_channels,1))
processed_data_noOffset = APclip - processed_data_mean*processed_data
#print(torch.min(processed_data_noOffset))


#move data from [n_channels,n_samples] to [x_array,y_array,n_samples,RGB]
array_in = torch.zeros((array[0],array[1],nS,3),dtype=torch.double)
array_in[idx_dim0,idx_dim1,:,0] = torch.einsum('ij->ji',(processed_data_noOffset,)).reshape(array[0]*array[1],1,nS)

array_in= array_in.numpy()
#plt.figure()
#plt.imshow(array_in[:,:,300,0])
#plt.show()
width = 32
hieght = 16 
channel = 1
 
fps = 20
sec = 5
 
# Syntax: VideoWriter_fourcc(c1, c2, c3, c4) # Concatenates 4 chars to a fourcc code
#  cv2.VideoWriter_fourcc('M','J','P','G') or cv2.VideoWriter_fourcc(*'MJPG)
 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # FourCC is a 4-byte code used to specify the video codec.
# A video codec is software or hardware that compresses and decompresses digital video. 
# In the context of video compression, codec is a portmanteau of encoder and decoder, 
# while a device that only compresses is typically called an encoder, and one that only 
# decompresses is a decoder. Source - Wikipedia
 
#Syntax: cv2.VideoWriter( filename, fourcc, fps, frameSize )
video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (width, hieght))
 
for frame_count in range(fps*sec):
    img = array_in
    img8 = (img/4).astype('uint8')
    #print(np.max(img8))
    #print(np.min(img8))
    video.write(img8)
    
video.release()


size = 720*16//9, 720
duration = 2
fps = 25
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for _ in range(fps * duration):
    data = np.random.randint(0, 256, size, dtype='uint8')
    print(np.shape(data))
    out.write(data)
out.release()

#compress 
