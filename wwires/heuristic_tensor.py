import torch
from gpu_memory import *
import numpy as np
import pdb
import scipy.io as sio

def heuristic_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda, P_info, collision = 32):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    
    #########################################################
    # Removed cuda device
    ramp = torch.linspace(v_rn, v_rp, (v_rp-v_rn)/lsb + 1, device = cuda).type(torch.int16)
#     ramp = torch.linspace(v_rn, v_rp, (v_rp-v_rn)/lsb + 1).type(torch.int16)
    #########################################################
    
    ramp = ramp.view(ramp.shape[0],1,1,1)

    #########################################################
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, (v_rp-v_rn)/lsb + 1, device = cuda).type(torch.int16)
#     ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, (v_rp-v_rn)/lsb + 1).type(torch.int16)
    #########################################################
    
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1)
    
#     pdb.set_trace()
    
    # create array of comparator outputs
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next)
    # extract row and column tensors for all samples and ramp values
    row = comp.any(dim=2)
    col = comp.any(dim=1)
    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts 
    
    #########################################################
    # old valid also takes into account if nothing is firing, new valid doesn't and takes into account only is something
#     valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = (torch.eq(torch.sum(row,dim=1),1) | torch.eq(torch.sum(col,dim=1),1))
    
#     print(collision)
    valid_collision = (torch.le(torch.sum(row,dim=1), collision) & 
                       torch.le(torch.sum(col,dim=1), collision) & 
                       torch.ne(torch.sum(row,dim=1), 0))
    
    #########################################################
        
    valid = torch.einsum("ij->ji", (valid,))
    valid_collision = torch.einsum("ij->ji", (valid_collision,))

    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0], row.shape[1], row.shape[2], 1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    
#     pdb.set_trace()
    valid = valid.view(valid.shape[0], valid.shape[1], 1, 1)
    valid_collision = valid_collision.view(valid_collision.shape[0], valid_collision.shape[1], 1, 1)
    array_out = torch.mul(array_out, valid_collision.float())
    
    #########################################################
    # Compression:
    
#     print(torch.sum(array_out)/(num_bits*array[0]*array[1]*samples_batch))

#     #  Naive:
#     row_avg = torch.sum(valid.byte().any(dim=3)).item()/(samples_batch)
#     col_avg = torch.sum(valid.byte().any(dim=2)).item()/(samples_batch)
    
    # Heuristic:
    row_avg = torch.sum(array_out.byte().any(dim=3)).item()/(samples_batch)
    col_avg = torch.sum(array_out.byte().any(dim=2)).item()/(samples_batch)
    bit_rate = row_avg*np.log(array[0])/np.log(2) + col_avg*np.log(array[1])/np.log(2)
    compression = (num_bits*array[0]*array[1])/bit_rate

#     print('Avg. Number of Active Rows = '+str(row_avg ))
#     print('Avg. Number of Active Cols = '+str(col_avg ))  
#     print('Avg. Bit Rate = '+str(bit_rate) )
#     print('Avg. Compression = '+str(compression) )
    
#     pdb.set_trace()
    #########################################################
    
    #########################################################
    
#     pdb.set_trace()
    
#     P = torch.div(torch.ones((samples_batch, 1, array[0], array[1]), device = cuda),(array[0]*array[1]))
#     amp_set =  np.zeros((array[0], array[1], samples_batch), dtype='bool_')
    
    amp_index_order = torch.cat((torch.linspace(v_rn, -1, -v_rn/lsb, device = cuda).type(torch.int16),
                                 torch.linspace(v_rp, 0, v_rp/lsb+1, device = cuda).type(torch.int16)),0)
        
#     naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,ramp.shape[0],array[0],array[1])
    naive_valid = torch.mul(array_out, valid.float())
    naive_valid = naive_valid[:,:,idx_dim0,idx_dim1].view(samples_batch,ramp.shape[0],array[0]*array[1])
    
#     active_row = torch.mul(row,valid)
#     active_col = torch.mul(col,valid)
#     active_out = torch.matmul(torch.mul(row,valid).float(), torch.mul(col,valid).float())
#     active_out = torch.mul(array_out, valid.float()).cuda()
    
#     pdb.set_trace()
    
    with torch.no_grad():
        m = torch.nn.Conv2d(1,1,3).cuda()
        m.padding = 1
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    
##################################################################################################################   
############################################ New stuff: #########################################################

    del row, col
    torch.cuda.empty_cache()
    
    P = torch.div(torch.ones((samples_batch, array[0]*array[1]), device = cuda),(array[0]*array[1]))
    
    active_out = torch.mul(array_out, valid.float()).cuda()
    array_out = array_out[:,:,idx_dim0,idx_dim1].view(samples_batch,ramp.shape[0],array[0]*array[1])
            
#     decoded_data = torch.zeros((samples_batch, array[0]*array[1]), device = cuda)
    decoded_data = torch.einsum("ijk,j->ik", (naive_valid,ramp.view(ramp.shape[0]).float()))
    decoded_data_naive = torch.einsum("ijk,j->ik", (naive_valid,ramp.view(ramp.shape[0]).float()))
        
#     del naive_valid
    torch.cuda.empty_cache()
    
    for amp_index, amp in enumerate(amp_index_order):
        
        amp_to_rampindex = (amp - v_rn.item())/lsb
        
        increase_P = m(active_out[:,amp_to_rampindex,:,:].view(samples_batch, 1, array[0], array[1]))
#         increase_P[increase_P>0] = 1
        increase_P = increase_P[:,:,idx_dim0,idx_dim1].view(samples_batch,array[0]*array[1])
        
        P = torch.mul(P, torch.mul(increase_P, P_info['P_inc']) + torch.mul(1 - increase_P, P_info['P_dec']) )
        
#         print('Not Valid: Ramp = ' + str(amp.item()-v_rn.item()))
        
#         if amp == -10:
#             pdb.set_trace()

##############################################
        
#         all_data_P = torch.mul(array_out[:,amp_to_rampindex,:], P)
#         unresolved_data_P = torch.mul(all_data_P, torch.eq( decoded_data_naive, 0).float() )
        
#         max_new, assign_new = torch.topk(unresolved_data_P, P_info['r_firing'] ) 
        
# #         max_new, assign_new = torch.topk( torch.mul(array_out[:,amp_to_rampindex,:], P), P_info['r_firing'] ) 
        
# #         samples_to_assign = torch.mul( torch.gt(max_new,0), torch.eq( decoded_data_naive, 0) )
    
#         for i in range(P_info['r_firing']):
# #             pdb.set_trace()
#             decoded_data[torch.arange(samples_batch).long(), assign_new[:,i]] = torch.mul(
#                 (1-naive_valid[torch.arange(samples_batch).long(),amp_to_rampindex,assign_new[:,i]]).float(), amp.item())
        
##############################################

        for j in range(samples_batch):
            if valid[j,amp_to_rampindex,0,0].item() == 1:
                continue
            
#             print('Not Valid: Ramp = ' + str(amp.item()-v_rn.item()) + ', Sample = ' + str(j))
                  
            max_new, assign_new = torch.topk( torch.mul(array_out[j,amp_to_rampindex,:], P[j,:]), P_info['r_firing'] )
            
#             pdb.set_trace()
#             r1 = (torch.mul(torch.eq(decoded_data_naive[j,assign_new],0).int(),amp.item()))
#             r2 = (1-torch.eq(max_new,0)).int()
#             decoded_data[j,assign_new] = torch.mul(r1,r2).float()
                                                               
            for i in range(P_info['r_firing']):

                if (max_new[i].item() == 0) or (decoded_data_naive[j,assign_new[i]].item() != 0): 
                    continue
                
#                 print('Decoded data should be 0: ' + str(decoded_data[j, assign_new[i]].item()))

                decoded_data[j, assign_new[i]] = amp.item() 
#                 P[j, assign_new[i]] = torch.mul(P[j, assign_new[i]], P_info['P_inc'])
##############################################

#     decoded_data += torch.einsum("ijk,j->ik", (naive_valid,ramp.view(ramp.shape[0]).float()))
    
    del active_out, array_out, P, increase_P, valid, valid_collision #all_data_P, unresolved_data_P #, decoded_data_naive
    torch.cuda.empty_cache()
    
    return decoded_data, decoded_data_naive, compression, bit_rate
##################################################################################################################   


##################################################################################################################   
############################################ Old stuff: #########################################################
# #     increase_P = m(active_out.view(samples_batch*ramp.shape[0], 1, array[0], array[1])).view(samples_batch, 
# #                                                                                              ramp.shape[0], array[0], array[1])
    
# #     P = torch.mul(P, torch.mul(increase_P, P_info['P_inc']) + torch.mul(1 - increase_P, P_info['P_dec']) ) 
    
# #     del increase_P    
# #     decoded_data = torch.zeros(samples_batch, array[0]*array[1], device = cuda)
# #     decoded_data_naive = torch.zeros(samples_batch, array[0]*array[1], device = cuda)
#     decoded_data = torch.einsum("ijk,j->ik", (naive_valid,ramp.view(ramp.shape[0]).float()))
#     decoded_data_naive = torch.einsum("ijk,j->ik", (naive_valid,ramp.view(ramp.shape[0]).float()))
    
#     for amp_index, amp in enumerate(amp_index_order):
        
#         increase_P = m(active_out[:,amp-v_rn.item(),:,:].view(samples_batch, 1, array[0], array[1]))
        
#         increase_P[increase_P>0] = 1
        
#         P = torch.mul(P, torch.mul(increase_P, P_info['P_inc']) + torch.mul(1 - increase_P, P_info['P_dec']) )
        
#         unresolved_P = torch.mul( (1-valid[:,amp-v_rn.item()].float()).view(samples_batch,1,1,1), 
#                                  P ).view(samples_batch, array[0], array[1])
    
#         unresolved_P = unresolved_P[:, idx_dim0, idx_dim1].view(samples_batch, array[0]*array[1])
        
#         _, assign_new = torch.topk(unresolved_P, P_info['r_firing'])
        
# #         decoded_data = naive_valid[:, amp-v_rn.item(),:] + torch.mul(torch.gather(unresolved_P, dim = 1, 
# #                                                                                   index =assign_new),amp.item())

# #         decoded_data = torch.add(decoded_data, torch.mul(naive_valid[:, amp-v_rn.item(),:], amp.item()))
# #         decoded_data_naive = torch.add(decoded_data_naive, torch.mul(naive_valid[:, amp-v_rn.item(),:], amp.item()))
        
# #         if amp_index == 1 or amp_index == 38:
# #             pdb.set_trace()
# #         decoded_data = torch.add(decoded_data, torch.mul(naive_valid[:, amp-v_rn.item(),:], amp.item()))
# #         decoded_data_naive = torch.add(decoded_data_naive, torch.mul(naive_valid[:, amp-v_rn.item(),:], amp.item()))
        
# #         for j in range(samples_batch):
# # #             pdb.set_trace()
# #             if valid[j,0,0].item()==1:
# #                 continue
# #             for i in range(P_info['r_firing']):
# #                 decoded_data[j, assign_new[j,i]] = decode_data[j, ]#0#amp.item() 
          
#         for i in range(P_info['r_firing']):
#             for j in range(samples_batch):
#                 if valid[j,0,0].item() == 1:
#                     continue
#                 if (naive_valid[j,amp-v_rn.item(), assign_new[j,i]].item() == 1):
#                     decoded_data[j, assign_new[j,i]] = amp.item() 
    
# #         for i in range(P_info['r_firing']):
            
# #             decoded_data[torch.arange(0, samples_batch).long(), assign_new[:,i]] = torch.mul(
# #                 (1-valid[torch.arange(0, samples_batch).long(), amp-v_rn.item()].float()).view(samples_batch), amp.item()) 
            
# #             decoded_data[torch.arange(0, samples_batch).long(), assign_new[:,i]] = torch.mul(
# #                 (1-naive_valid[torch.arange(0, samples_batch).long(), amp-v_rn.item(),assign_new[:,i]]), amp.item()) 
            
# #             decoded_data = torch.mul((1-naive_valid[:, amp-v_rn.item(),:]), decoded_data)
            
# #             decoded_data[torch.arange(0, samples_batch).long(), assign_new[:,i]] = amp.item() 
# #             decoded_data = torch.add(decoded_data, torch.mul(
# #                 decoded_data[torch.arange(0, samples_batch).long(), assign_new[:,i]], amp.item()))
        
#     return decoded_data, decoded_data_naive
##################################################################################################################


    #########################################################
#     # free some space
#     del row, col
#     torch.cuda.empty_cache()
#     # go back to all channels in one vector
#     naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,ramp.shape[0],array[0]*array[1])
#     # reshape valid for torch.mul
#     valid = valid.view(valid.shape[0], valid.shape[1], 1)
#     # use torch.mul to get rid of invalid arguments
#     naive_valid = torch.mul(naive_out,valid.float())
#     # free some space
#     del naive_out
#     torch.cuda.empty_cache()
#     # reshape ramp for torch.einsum
#     ramp = ramp.view(ramp.shape[0])
#     # use torch.einsum to sum across all ramp values and get decoded output
#     naive_decoded = torch.einsum("ijk,j->ik", (naive_valid,ramp.float()))
#     return naive_decoded
            


#     def decoding_rect(decoding_input, P, P_inc, P_dec, t, cuda):
#         P = torch.ones((16,32),dtype='float32')/512

#         #amp_index_order = np.linspace(-2**(B-1),2**(B-1)-1,2**B,dtype='int32')

#         amp_index_order = torch.append(torch.linspace(-2**(B-1),-1,2**(B-1),dtype='int32'),
#                                     torch.linspace(2**(B-1)-1,0,2**(B-1),dtype='int32'))

#         row,col = decoding_input
#         decoding_output = torch.zeros((16,32),dtype='float32')
#         decoding_naive_output = torch.zeros((16,32),dtype='float32')
#         amp_set =  torch.zeros((16,32),dtype='float32')

#         print('Curr Time Sample= ' + str(t))

#         for amp_index,amp in enumerate(amp_index_order):
#             #print('Curr Time Sample= ' + str(t) + ', Curr Amp Index= ' + str(amp_index))

#             active_row = [j for j,v in enumerate(row[:,amp+2**(B-1)]) if v > 0]
#             active_col = [j for j,v in enumerate(col[:,amp+2**(B-1)]) if v > 0]

#             if len(active_row)==0 and len(active_col)==0:
#                 continue
#             elif len(active_row)==1 and len(active_col)==1:
#                 decoding_output[active_row,active_col] = amp
#                 decoding_naive_output[active_row,active_col] = amp

#                 amp_set[active_row,active_col] = 1

#                 P[active_row[0]-1:active_row[0]+2,active_col[0]-1:active_col[0]+2] *= P_inc

#             else:
#                 #pdb.set_trace()
#                 active_col_v, active_row_v = torch.meshgrid(active_col,active_row)
#                 P_reshaped = P[active_row_v,active_col_v].reshape(-1,len(active_row)*len(active_col))
#                 #pdb.set_trace()
#                 firing_set = torch.vstack((torch.argsort(P_reshaped)[0,-r_firing:]//32, \
#                                              torch.argsort(P_reshaped)[0,-r_firing:]%32)).T

#                 for uf_ind,uf in enumerate(firing_set):
#                     #if decoding_output[firing_set[uf_ind][0],firing_set[uf_ind][1]]==0:
#                     if amp_set[firing_set[uf_ind][0],firing_set[uf_ind][1]] == 0:
#                         decoding_output[firing_set[uf_ind][0],firing_set[uf_ind][1]]=amp
#                         P[firing_set[uf_ind][0],firing_set[uf_ind][1]] *= P_inc


#             P *= P_dec

#         return decoding_output,decoding_naive_output,P