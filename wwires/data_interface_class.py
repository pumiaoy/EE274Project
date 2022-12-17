import numpy as np
import h5py
import time
import struct
import calendar
import os

class io_object:
    """
    Class to handle the input and output of bin files. 
    For input assumes bin file has been converted to a mat file using matlab functions.
    For output, writes the bin file which can be post-processed normally using smash and imported in vision.
    
    Has methods to:
    A) Read input data in inp_file of size time_window in batches of chunk_size.
    B) Generate header for writing processed data in a standard experiemnt format bin file. 
       *HACK: right now hardcoded for standard 512 litke array*
    C) Generate processed data in compressed format. This is standard experiment format for writing data in bin file. 
       It compresses 2 X uint16 format recorded data having 12 bits actual data into 3 X uint8 format data.
    D) Write bin file in save_path as bin_filename. Can choose to overwrite existing file. Appends if written in bacthes.
    """    
    def __init__(self, time_window, chunk_size, inp_file='[]', output_data_batch='', 
                     save_path='[]', bin_filename='data000000.bin', num_bits = 12, fs = 20000, overwrite = False): #now vref is actually 10bits range what needs to change here?
        """
        Initializes data interface class.
        
        Args:
            time_window: time window for which input data needs to be read in seconds
            chunk_size: number of samples to be processed in single batch
            inp_file: path to input mat file to be read (give complete path). Expected to be an array of size (self.num_channels, self.chunk_size) with saved variable name 'newData'
            output_data_batch: batch of output data in numpy array format to be saved. Expected to be of size (self.num_channels, self.chunk_size)
            save_path: path to save output bin file. If it doesn't exist, a new folder is created.
            bin_filename: name of the bin file to be saved
            num_bits: number of bits assumed during processing of the input data
            fs: sampling frequency
            overwrite: whether to overwrite the bin file if the save_path+bin_filename already exists. Checks only for initial batch of data being saved allowing appending of data in batches to the saved file even if overwrite is False.
        """
        
        self.inp_file = inp_file
        self.time_window = time_window
        self.chunk_size = chunk_size
        self.output_data_batch = output_data_batch
        self.save_path = save_path
        self.bin_filename = bin_filename
        self.num_bits = num_bits
        self.fs = fs
        self.overwrite = overwrite
        
        self.file = h5py.File(self.inp_file)
        self.num_channels = 512 + 1 # +1 for TTL. *HACK: hardcoded for 512 Litke array*
        self.num_samples = int(self.fs*self.time_window) # Calculate total number of samples to be processed
        self.num_batches = int((self.num_samples-1)/self.chunk_size) + 1 # Number of batches. Can be accessed from outside.
        
    def read_batch(self, batch_num, print_every):
        """
        Get batch of input data.
        Also, extracts trigger info for generating header.
        Args:
            batch_num: current batch number. If batch_number is 0 we need to be careful while writing the bin file of the processed data because we will have to merge header, check for overwrite etc. 
            print_every: prints every print_every batch to keep track of running code
        """
        
            # Checks if available number of samples left from total samples is atleast chunk_size. If yes: grab the next chunk_size samples in self.raw_data_batch. Else: grab the remaining samples.                 
        if self.chunk_size*(batch_num+1) <= self.num_samples:
            self.raw_data_batch = np.array(self.file['newData'][:,self.chunk_size*batch_num:self.chunk_size*(batch_num+1)])
        else:
            self.raw_data_batch = np.array(self.file['newData'][:,self.chunk_size*batch_num:self.num_samples])

        # If first batch, extract trigger properties and set the self.run_header flag to be true so that the header is added to bin file before addding processed data while writing. Note htat 0th channel is trigger channel in raw data.
        if batch_num == 0:
            print('Shape of a Raw Data Batch = ' + str(self.raw_data_batch.shape))

            # Extract first trigger delay time and duration to be written in header:
            #trig_times = np.where(self.raw_data_batch[0,:]<0)[0]

            trig_times = np.where(self.raw_data_batch[0,:]<0)[0]
            self.first_trig_delay = trig_times[0]/self.fs # In seconds; Found Emperically
            self.first_trig_interval = 1 # In samples
            for trig_ind in range(len(trig_times)-1):
                if trig_times[trig_ind+1] - trig_times[trig_ind] == 1:
                    self.first_trig_interval = self.first_trig_interval + 1
                else:
                    break
            self.first_trig_interval = self.first_trig_interval/self.fs # Convert to seconds
            self.run_header = True
        else:
            self.run_header = False

        if batch_num%print_every == 0:
            print('Processing: Batch Number = ' + str(batch_num))
            
    def generate_header(self):
        """
        Generates header for writing bin file of processed data. 
        *HACK: right now hardcoded for standard 512 litke array* 
        Refer to the documentation 'readme_binfile_writing.txt' for more details.
        """
        if self.run_header == True:
            user_header = b"modified"
            #header_len = 2*25 + 2*len(user_header)
            header_len = 132 #* HACK: HARDCODED FOR NOW BUT CHANGE IT!*
            self.header = b''

            self.header += struct.pack(">I",0) #tag 0 for header length
            self.header += struct.pack(">I",4) #tag payload is 4 bytes
            self.header += struct.pack(">I",header_len) #tag payload is 4 bytes

            self.header += struct.pack(">I",1) #tag 1 for time
            self.header += struct.pack(">I",12) #tag payload is 12 bytes
            self.header += struct.pack(">I",1904) #tag payload is 4 bytes
            unix_time = calendar.timegm(time.gmtime())
            self.header += struct.pack(">Q",unix_time) #tag payload is 4 bytes

            self.header += struct.pack(">I",2) 
            self.header += struct.pack(">I", len(user_header))
            self.header += user_header

            self.header += struct.pack(">I",3) #Compression flag
            self.header += struct.pack(">I",4)
            self.header += struct.pack(">I",1) #compression!

            self.header += struct.pack(">I",4) #array type
            self.header += struct.pack(">I",8)
            self.header += struct.pack(">I",self.num_channels)
            self.header += struct.pack(">I",504) 

            self.header += struct.pack(">I",5) #frequency in Hz
            self.header += struct.pack(">I",4)
            self.header += struct.pack(">I",self.fs)

            self.header += struct.pack(">I",6) # trigger params
            self.header += struct.pack(">I",8)
            self.header += struct.pack(">I",int(self.first_trig_delay*1000000)) # First Trigger Delay in seconds * 1000000
            self.header += struct.pack(">I",int(self.first_trig_interval*1000000)) # First Trigger Interval in seconds * 1000000

            self.header += struct.pack(">I",7) #dataset identifier
            self.header += struct.pack(">I",len(user_header))
            self.header += user_header

            self.header += struct.pack(">I",499) #endtag
            self.header += struct.pack(">I",4)
            self.header += struct.pack(">I",self.num_samples)

            self.write_header = True
        else:
            self.write_header = False

    def generate_compressed_data(self):
        """
        Generates the processed data in compressed form to be stored in bin files.
        Reduces bin file size by reducing the redundant 4 bits in the typical uint16 format in which data is stored (Litke array samples data at 12 bits only!).
        It compresses 2 uint16 format data into 3 uint8 format bytes to be stored in bin file. This data is constructed in a specific manner. Refer to the documentation 'readme_binfile_writing.txt' for more details.
        """
        
        # Check if the output data batch is in right format and size (ttl added, shape etc)
        if type(self.output_data_batch) != np.ndarray or np.shape(self.output_data_batch)[0] != (self.num_channels):
                raise Exception('Output Data Batch is not specified properly!')
        else:    
            self.output_data_batch = self.output_data_batch.astype(np.int16)

            # Copy ttl data (triggers):
            sample_ttl = np.zeros((2,self.output_data_batch.shape[1]),dtype = 'uint8') 
            sample_ttl[0,:] = np.right_shift(np.bitwise_and(self.output_data_batch[0:1,:],0xFF00),8)
            sample_ttl[1,:] = np.bitwise_and(self.output_data_batch[0:1,:],0x00FF)

            # Compress recorded data:
            sample = np.zeros((self.output_data_batch.shape[0]-1,self.output_data_batch.shape[1]),dtype = 'uint16')
            sample = (self.output_data_batch[1:,:] + 2048).astype('uint16')
            self.sample_combined=np.zeros((int((self.output_data_batch.shape[0]-1)*1.5),self.output_data_batch.shape[1]),
                                          dtype='uint8')

            for first_of_two_channel in range(0,self.num_channels-1,2):
                self.sample_combined[int(first_of_two_channel*1.5),:] = np.right_shift(sample[first_of_two_channel,:],4)
                self.sample_combined[int(first_of_two_channel*1.5)+1,:] = np.left_shift(np.bitwise_and(
                    sample[first_of_two_channel,:],0x000F),4) + np.right_shift(sample[first_of_two_channel+1,:],8)
                self.sample_combined[int(first_of_two_channel*1.5)+2,:] = np.bitwise_and(sample[first_of_two_channel+1,:],0x00FF)

            # Combine ttl and recorded data to be written into bin file:
            self.sample_combined = np.concatenate((sample_ttl, self.sample_combined), axis=0)

    def write_bin_file(self):
        """
        Writes a bin file in batches. Needs header and compressed data to be generated first. creates a dircetory if it doesn't exist and checks for overwrite. Appends the datawritten in batches. Adds header to the beginning of bin file. 
        """
        
        # If no directory exists at path, make one.
        if not os.path.exists(self.save_path):
            print('Directory did not exist. Making a new one...')
            os.makedirs(self.save_path)
                        
        bin_file_path = os.path.join(self.save_path + '/' + self.bin_filename)
        
        # Check in the initial batch if the path exists (note: write_header is true only for 0th batch, check read_batch method)
        if self.write_header == True and os.path.exists(bin_file_path):
            if self.overwrite:
                os.remove(bin_file_path)
            else:
                raise Exception('Save bin file path and name already exists! Change one.')
       
        # Make sure binfile is being appended for batches and not overwritten
        bin_file_pointer = open(bin_file_path, "ab+")
        
        # Write header for initial batch
        if self.write_header == True:
            bin_file_pointer.write(self.header)
        
        # Write data (Note: since data is in amtrix forma, it needs to be traversed in a specific manner for the bin file to be recognized correctly as described in 'readme_binfile_writing.txt'. That's why tobytes('F') is needed)
        bin_file_pointer.write(self.sample_combined.tobytes('F')) 
        
        bin_file_pointer.close()
