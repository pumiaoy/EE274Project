# Author: Pumiao Yan 
#verify data fully preserved when only one channel is active

from naive_multiwire_tensor import *
import h5py
import mat73
import pytest
import numpy as np
import torch
import time
import scipy.io as sio 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_collision1():
	time_window = 1
	#initialize input
	array = [16,32]
	pos_lin = sio.loadmat('posLin.mat')['posLin'].astype('int') - 1 #-1 since bringing matlab indices to python
	pos_log = sio.loadmat('posLog.mat')['posLog'].astype('int') - 1 #-1 since bringing matlab indices to python
	n_channels = pos_lin.shape[0]
	test_baches = 40 # number of baches to be tested
	chunk_size =500 # chunk of samples to be processed simultaneously
	num_bits = [10]
	n_wires = [1]
	ovr = True
	data_path ='../data/test/'
	strategy = 'naive_2ramp/' +'3Proj/'
	subset = '2'
	save_path = data_path + strategy + str(num_bits) + 'b_' + str(n_wires) + 'w/' + subset
	inp_file = '../data/test/test_data2.mat'

	#run main_naive code
	for b in range(len(num_bits)):
		for w in range(len(n_wires)):
			print('Running naive decoder with ' + str(num_bits[b]) + ' bits and ' + str(n_wires[w]) + " wires")
			Output = naive_multiwire_tensor(inp_file, save_path, time_window, chunk_size*test_baches, num_bits[b], n_wires[w], ovr)
	Input = h5py.File(inp_file)
	Input = Input.get('newData')
	assert np.array_equal(Input, Output)
	
	

#compare output and input for signal integrity
#calculate enob
#print
#check result
#assert()