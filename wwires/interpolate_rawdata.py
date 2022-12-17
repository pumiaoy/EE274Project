import torch

def interpolate_rawdata(decoded_data,samples_batch,cuda,sigma):
    #decoded_data = torch.tensor([[1, 0, 0, 2, 4, 0, 1],[1, 0, 0, 2, 4, 0, 1],[1, 0, 0, 2, 4, 0, 1]]).type(torch.FloatTensor)
    #decoded_data = torch.einsum('ij->ji', (decoded_data,)).to(cuda)
    #samples_batch= decoded_data.shape[0]
    interpolated_data = torch.zeros((samples_batch, decoded_data.shape[1]), device = cuda)
    
    logic_matrix_zero = torch.eq(decoded_data[1:samples_batch-1,:],0).type(torch.FloatTensor).to(cuda)
    perm_previous = torch.arange(0,samples_batch-2).type(torch.LongTensor).to(cuda)
    perm_next = torch.arange(2,samples_batch).type(torch.LongTensor).to(cuda)
    interpolated_data[1:samples_batch-1,:] = decoded_data[1:samples_batch-1,:] + \
                                        logic_matrix_zero*(torch.randn_like(logic_matrix_zero)*sigma+(decoded_data[perm_previous,:]+decoded_data[perm_next,:])/2)
    return interpolated_data

