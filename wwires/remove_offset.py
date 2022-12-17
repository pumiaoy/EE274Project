import torch

def remove_offset(raw_data):

    raw_data_pp_mean = raw_data - torch.mean(raw_data, 1, True)
    
    return raw_data_pp_mean.type(torch.int16)

