import torch, math
from gpu_memory import *
import matplotlib.pyplot as plt

def rect_naive_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in).item()//lsb)*lsb - lsb
    v_rp = (torch.max(array_in).item()//lsb)*lsb + lsb

    num_of_ramp_steps = int((v_rp-v_rn)/lsb + 1)

    ramp = torch.linspace(v_rn, v_rp, num_of_ramp_steps, device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, num_of_ramp_steps, device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1)
    # create array of comparator outputs
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next)
    # extract row and column tensors for all samples and ramp values

    row = comp.any(dim=2)
    col = comp.any(dim=1)
    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    vzeros = (torch.le(torch.sum(row,dim=1),0) | torch.le(torch.sum(col,dim=1),1))

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
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,ramp.shape[0],array[0]*array[1])
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)

    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out,valid.float())
    # free some space
    del naive_out

    torch.cuda.empty_cache()
    # reshape ramp for torch.einsum
    ramp = ramp.view(ramp.shape[0])
    # use torch.einsum to sum across all ramp values and get decoded output
    naive_decoded = torch.einsum("ijk,j->ik", (naive_valid,ramp.float()))
    return naive_decoded


def rect_naive_2proj_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in).item()//lsb)*lsb - lsb
    v_rp = (torch.max(array_in).item()//lsb)*lsb + lsb

    num_of_ramp_steps = int((v_rp-v_rn)/lsb + 1)

    ramp = torch.linspace(v_rn, v_rp, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1) #reshape
    # create array of comparator outputs
  
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next) #quantization
    # extract row and column tensors for all samples and ramp values

    num_rlevels = comp.size()[0]
    num_rows    = comp.size()[1]
    num_cols    = comp.size()[2]
    num_batches = comp.size()[3]

    row = comp.any(dim=2)
    col = comp.any(dim=1)

    #extract 3rd projection tensor
    #print(comp.cpu().numpy())
    #digl = projection_onto_right_diagonal(comp.type(torch.uint8))
    #print(digl.cpu().numpy())
    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    #decoder starts here:
    # 0. Noise toss case: if all projections has a threshold value of wires active--> Noise range 6 rows cols and diags

    #These maps are used to select the batch and ramp using the comparison mask that is processible by a simple decoder
    ramp_idx_map  = torch.arange(0,num_rlevels, 1, dtype=torch.long, device=cuda).repeat(num_batches,1).T
    batch_idx_map =  torch.arange(0, num_batches, 1, dtype=torch.long, device=cuda).T.repeat(num_rlevels,1)
    #cond_all_eq_two = torch.le(torch.sum(row, dim=1), 2) & torch.le(torch.sum(col, dim=1), 2) & torch.le(torch.sum(digl, dim=1), 2)
    cond_any_le_one = torch.eq(torch.sum(row, dim=1), 1) | torch.eq(torch.sum(col, dim=1), 1) #| torch.eq(torch.sum(digl, dim=1), 1)

    cond_req1 = torch.eq(torch.sum(row, dim=1), 1)
    cond_ceq1 = torch.eq(torch.sum(col, dim=1), 1)
    #cond_deq1 = torch.eq(torch.sum(digl, dim=1), 1)

    cond_req2 = torch.eq(torch.sum(row, dim=1), 2)
    cond_ceq2 = torch.eq(torch.sum(col, dim=1), 2)
    #cond_deq2 = torch.eq(torch.sum(digl, dim=1), 2)

    oned_select_mask = cond_any_le_one 
    zero_select_mask = torch.eq(torch.sum(row, dim=1), 0) & torch.eq(torch.sum(col, dim=1), 0) #& torch.eq(torch.sum(digl, dim=1), 0)

    valid_ramp_indices = torch.masked_select(ramp_idx_map, oned_select_mask)
    valid_batch_indices = torch.masked_select(batch_idx_map, oned_select_mask)

    valid_col = col[valid_ramp_indices,:, valid_batch_indices]
    valid_row = row[valid_ramp_indices,:, valid_batch_indices]
    #valid_dlg = digl[valid_ramp_indices,:, valid_batch_indices]


    #bit_vld_col = (torch.sum(torch.eq(torch.sum(valid_col, dim=1), 1))*6 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 2))*11 + torch.sum(torch.gt(torch.sum(valid_col, dim=1), 2))*33)
    #bit_vld_dlg = (torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 1))*6 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 2))*11 + torch.sum(torch.gt(torch.sum(valid_dlg, dim=1), 2))*33)
    #bit_vld_row = (torch.sum(torch.eq(torch.sum(valid_row, dim=1), 1))*5 + torch.sum(torch.eq(torch.sum(valid_row, dim=1), 2))*9 + torch.sum(torch.gt(torch.sum(valid_row, dim=1), 2))*17)

    bit_vld_col = (torch.sum(torch.eq(torch.sum(valid_col, dim=1), 1))*6 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 2))*11 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 3))*17 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 4))*23 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 5))*28 + torch.sum(torch.gt(torch.sum(valid_col, dim=1), 5))*33)
    #bit_vld_dlg = (torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 2))*11 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 3))*17 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 4))*23 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 5))*28 + torch.sum(torch.gt(torch.sum(valid_dlg, dim=1), 5))*33) 
    bit_vld_row = (torch.sum(torch.eq(torch.sum(valid_row, dim=1), 1))*5 + torch.sum(torch.eq(torch.sum(valid_row, dim=1), 2))*9 + torch.sum(torch.gt(torch.sum(valid_row, dim=1), 2))*17)

    bit_vld_tot = (bit_vld_col+bit_vld_row).float()/(num_batches)

    compression_rate = (num_rows*num_cols)*num_bits/bit_vld_tot
    print(bit_vld_tot)
    print(compression_rate)

    xy_shadow = torch.einsum("bi,bj->bij",(valid_row.float(), valid_col.float()))

    #diag_basis = projection_onto_right_basis((num_rows, num_cols))
    #reordered_diag_basis = diag_basis.permute(1,0,2)

    #reordered_diag_shadow = torch.matmul(valid_dlg.float(), reordered_diag_basis.float())
    #diag_shadow = torch.einsum("ijk->jik", reordered_diag_shadow)

    valid_array_out = xy_shadow

    del  xy_shadow, #reordered_diag_shadow, reordered_diag_basis, diag_basis
    del valid_col, valid_row, ramp_idx_map, batch_idx_map, oned_select_mask
    torch.cuda.empty_cache()


    valid_naive_out = torch.zeros((num_rlevels, num_rows, num_cols, num_batches), device=cuda)
    valid_naive_out[valid_ramp_indices, :, :, valid_batch_indices] = valid_array_out

    reshape_naive_out = torch.einsum('hijk->kijh', valid_naive_out)
    naive_decoded = torch.matmul(reshape_naive_out, ramp.view(ramp.shape[0]).float())


    flat_naive_decoded = naive_decoded[:,idx_dim0, idx_dim1].view(samples_batch,array[0]*array[1])
    del naive_decoded, reshape_naive_out, valid_naive_out
    torch.cuda.empty_cache()



    #    
#
    #if torch.le(torch.sum(row,dim=1),6) & torch.le(torch.sum(col,dim=1),6) & torch.le(torch.sum(digl,dim=1),6):
    #    
    #    # 1.If only one projection (simplest decoder)
    #    if valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1)| torch.le(torch.sum(digl,dim=1),1)):
    #
    #    # 2.If no more than two wires active in any projection (needing third projection info)
    #    elif (torch.le(torch.sum(row,dim=1),1) and torch.le(torch.sum(col,dim=1),1) and torch.le(torch.sum(digl,dim=1),1)):
    #        jjaladf
#
    #        else: # 3. complecated reconstruction case (shadow logic)

    return flat_naive_decoded


def rect_naive_3proj_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    def projection_onto_right_diagonal(comp_mat):
        #Split apart array shape description to make algorithm cleaner to read
        RLevels = comp_mat.size()[0]
        N =  comp_mat.size()[1]
        M =  comp_mat.size()[2]
        BatchSize =  comp_mat.size()[3]
        #Assert fatties only
        assert(N <= M)

        #Create new tensor vector for storing the output
        diag_or = torch.zeros([RLevels, M, BatchSize], dtype=torch.uint8).cuda()
        comp_mat = torch.einsum("lijk->lkij", comp_mat)
        #Process straight diags and concat diags separately
        boundary = M - N + 1
        for ii in range(M):
            if ii < boundary:
                diag_or[:,ii,:] = torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1).any(dim=2)
            else:
                diag_or[:,ii,:] = torch.cat((torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1), torch.diagonal(comp_mat, offset=ii-M, dim1=-2, dim2=-1)),-1).any(dim=2)
        return diag_or

    def projection_onto_right_basis(dimensions):
        N = dimensions[0]
        M = dimensions[1]

        def rect_diag(inp_tensor, diagonal=0):
            temp_diag = torch.diag(inp_tensor, diagonal=diagonal).type(torch.uint8).cuda()
            slice_diag = temp_diag[0:N,0:M]
            del temp_diag
            torch.cuda.empty_cache()

            return slice_diag

        assert(N<=M)

        diag_basis = torch.zeros([M, N, M], dtype=torch.uint8).cuda()
        boundary = M-N+1

        for ii in range(M):
            if ii < boundary:
                diag_basis[ii,:,:] = rect_diag(torch.ones(M), diagonal=ii)
            else:
                diag_basis[ii,:,:] = rect_diag(torch.ones(M), diagonal=ii)
                diag_basis[ii,:,:] += rect_diag(torch.ones(M), diagonal=ii-M)
        return diag_basis


        # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in).item()//lsb)*lsb - lsb
    v_rp = (torch.max(array_in).item()//lsb)*lsb + lsb

    num_of_ramp_steps = int((v_rp-v_rn)/lsb + 1)

    ramp = torch.linspace(v_rn, v_rp, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1) #reshape
    # create array of comparator outputs
  
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next) #quantization
    # extract row and column tensors for all samples and ramp values

    num_rlevels = comp.size()[0]
    num_rows    = comp.size()[1]
    num_cols    = comp.size()[2]
    num_batches = comp.size()[3]

    row = comp.any(dim=2)
    col = comp.any(dim=1)

    #extract 3rd projection tensor
    #print(comp.cpu().numpy())
    digl = projection_onto_right_diagonal(comp.type(torch.uint8))
    #print(digl.cpu().numpy())
    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    #decoder starts here:
    # 0. Noise toss case: if all projections has a threshold value of wires active--> Noise range 6 rows cols and diags

    #These maps are used to select the batch and ramp using the comparison mask that is processible by a simple decoder
    ramp_idx_map  = torch.arange(0,num_rlevels, 1, dtype=torch.long, device=cuda).repeat(num_batches,1).T
    batch_idx_map =  torch.arange(0, num_batches, 1, dtype=torch.long, device=cuda).T.repeat(num_rlevels,1)
    cond_all_eq_two = torch.le(torch.sum(row, dim=1), 2) & torch.le(torch.sum(col, dim=1), 2) & torch.le(torch.sum(digl, dim=1), 2)
    cond_any_le_one = torch.eq(torch.sum(row, dim=1), 1) | torch.eq(torch.sum(col, dim=1), 1) | torch.eq(torch.sum(digl, dim=1), 1)

    cond_req1 = torch.eq(torch.sum(row, dim=1), 1)
    cond_ceq1 = torch.eq(torch.sum(col, dim=1), 1)
    cond_deq1 = torch.eq(torch.sum(digl, dim=1), 1)

    cond_req2 = torch.eq(torch.sum(row, dim=1), 2)
    cond_ceq2 = torch.eq(torch.sum(col, dim=1), 2)
    cond_deq2 = torch.eq(torch.sum(digl, dim=1), 2)

    oned_select_mask = cond_any_le_one | cond_all_eq_two
    zero_select_mask = torch.eq(torch.sum(row, dim=1), 0) & torch.eq(torch.sum(col, dim=1), 0) & torch.eq(torch.sum(digl, dim=1), 0)

    valid_ramp_indices = torch.masked_select(ramp_idx_map, oned_select_mask)
    valid_batch_indices = torch.masked_select(batch_idx_map, oned_select_mask)

    valid_col = col[valid_ramp_indices,:, valid_batch_indices]
    valid_row = row[valid_ramp_indices,:, valid_batch_indices]
    valid_dlg = digl[valid_ramp_indices,:, valid_batch_indices]


    bit_vld_col = (torch.sum(torch.eq(torch.sum(valid_col, dim=1), 1))*5 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 2))*10 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 3))*15 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 4))*20 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 5))*25 + torch.sum(torch.gt(torch.sum(valid_col, dim=1), 5))*32)
    bit_vld_dlg = (torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 1))*0 +torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 2))*10 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 3))*15 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 4))*20 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 5))*25 + torch.sum(torch.gt(torch.sum(valid_dlg, dim=1), 5))*32) 
    bit_vld_row = (torch.sum(torch.eq(torch.sum(valid_row, dim=1), 1))*4 + torch.sum(torch.eq(torch.sum(valid_row, dim=1), 2))*8 + torch.sum(torch.gt(torch.sum(valid_row, dim=1), 2))*12)

    bit_vld_tot = (bit_vld_col+bit_vld_row+bit_vld_dlg).float()/(num_batches)

    compression_rate = (num_rows*num_cols)*num_bits/bit_vld_tot
    print(bit_vld_tot)
    #print(compression_rate)

    xy_shadow = torch.einsum("bi,bj->bij",(valid_row.float(), valid_col.float()))

    diag_basis = projection_onto_right_basis((num_rows, num_cols))
    reordered_diag_basis = diag_basis.permute(1,0,2)

    reordered_diag_shadow = torch.matmul(valid_dlg.float(), reordered_diag_basis.float())
    diag_shadow = torch.einsum("ijk->jik", reordered_diag_shadow)

    valid_array_out = torch.einsum('bij,bij->bij', (diag_shadow, xy_shadow))

    del diag_shadow, xy_shadow, reordered_diag_shadow, reordered_diag_basis, diag_basis
    del valid_col, valid_row, valid_dlg, ramp_idx_map, batch_idx_map, oned_select_mask
    torch.cuda.empty_cache()


    valid_naive_out = torch.zeros((num_rlevels, num_rows, num_cols, num_batches), device=cuda)
    valid_naive_out[valid_ramp_indices, :, :, valid_batch_indices] = valid_array_out

    reshape_naive_out = torch.einsum('hijk->kijh', valid_naive_out)
    naive_decoded = torch.matmul(reshape_naive_out, ramp.view(ramp.shape[0]).float())


    flat_naive_decoded = naive_decoded[:,idx_dim0, idx_dim1].view(samples_batch,array[0]*array[1])
    del naive_decoded, reshape_naive_out, valid_naive_out
    torch.cuda.empty_cache()



    #    
#
    #if torch.le(torch.sum(row,dim=1),6) & torch.le(torch.sum(col,dim=1),6) & torch.le(torch.sum(digl,dim=1),6):
    #    
    #    # 1.If only one projection (simplest decoder)
    #    if valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1)| torch.le(torch.sum(digl,dim=1),1)):
    #
    #    # 2.If no more than two wires active in any projection (needing third projection info)
    #    elif (torch.le(torch.sum(row,dim=1),1) and torch.le(torch.sum(col,dim=1),1) and torch.le(torch.sum(digl,dim=1),1)):
    #        jjaladf
#
    #        else: # 3. complecated reconstruction case (shadow logic)

    return flat_naive_decoded

def rect_naive_4proj_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    def projection_onto_right_diagonal(comp_mat):
        #Split apart array shape description to make algorithm cleaner to read
        RLevels = comp_mat.size()[0]
        N =  comp_mat.size()[1]
        M =  comp_mat.size()[2]
        BatchSize =  comp_mat.size()[3]
        #Assert fatties only
        assert(N <= M)

        #Create new tensor vector for storing the output
        diag_or = torch.zeros([RLevels, M, BatchSize], dtype=torch.uint8).cuda()
        comp_mat = torch.einsum("lijk->lkij", comp_mat)
        #Process straight diags and concat diags separately
        boundary = M - N + 1
        for ii in range(M):
            if ii < boundary:
                diag_or[:,ii,:] = torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1).any(dim=2)
            else:
                diag_or[:,ii,:] = torch.cat((torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1), torch.diagonal(comp_mat, offset=ii-M, dim1=-2, dim2=-1)),-1).any(dim=2)
        return diag_or

    def projection_onto_left_diagonal(comp_mat):
        #Split apart array shape description to make algorithm cleaner to read
        RLevels = comp_mat.size()[0]
        N =  comp_mat.size()[1]
        M =  comp_mat.size()[2]
        BatchSize =  comp_mat.size()[3]
        #Assert fatties only
        assert(N <= M)

        #Create new tensor vector for storing the output
        diag_or = torch.zeros([RLevels, M, BatchSize], dtype=torch.uint8).cuda()
        comp_mat = torch.flip(torch.einsum("lijk->lkij", comp_mat), dims=[-1])
        #Process straight diags and concat diags separately
        boundary = M - N + 1
        for ii in range(M):
            if ii < boundary:
                diag_or[:,ii,:] = torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1).any(dim=2)
            else:
                diag_or[:,ii,:] = torch.cat((torch.diagonal(comp_mat, offset=ii, dim1=-2, dim2=-1), torch.diagonal(comp_mat, offset=ii-M, dim1=-2, dim2=-1)),-1).any(dim=2)
        return diag_or

    def projection_onto_right_basis(dimensions):
        N = dimensions[0]
        M = dimensions[1]

        def rect_diag(inp_tensor, diagonal=0):
            temp_diag = torch.diag(inp_tensor, diagonal=diagonal).type(torch.uint8).cuda()
            slice_diag = temp_diag[0:N,0:M]
            del temp_diag
            torch.cuda.empty_cache()

            return slice_diag

        assert(N<=M)

        diag_basis = torch.zeros([M, N, M], dtype=torch.uint8).cuda()
        boundary = M-N+1

        for ii in range(M):
            if ii < boundary:
                diag_basis[ii,:,:] = rect_diag(torch.ones(M), diagonal=ii)
            else:
                diag_basis[ii,:,:] = rect_diag(torch.ones(M), diagonal=ii)
                diag_basis[ii,:,:] += rect_diag(torch.ones(M), diagonal=ii-M)
        return diag_basis

    def projection_onto_left_basis(dimensions):
        N = dimensions[0]
        M = dimensions[1]

        def rect_diag(inp_tensor, diagonal=0):
            temp_diag = torch.diag(inp_tensor, diagonal=diagonal).type(torch.uint8).cuda()
            slice_diag = temp_diag[0:N,0:M]
            del temp_diag
            torch.cuda.empty_cache()

            return slice_diag

        assert(N<=M)

        diag_basis = torch.zeros([M, N, M], dtype=torch.uint8).cuda()
        boundary = M-N+1

        for ii in range(M):
            if ii < boundary:
                diag_basis[ii,:,:] = torch.flip(rect_diag(torch.ones(M), diagonal=ii), dims=[1])
            else:
                diag_basis[ii,:,:] = torch.flip(rect_diag(torch.ones(M), diagonal=ii), dims=[1])
                diag_basis[ii,:,:] += torch.flip(rect_diag(torch.ones(M), diagonal=ii-M), dims=[1])
        return diag_basis

        # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in).item()//lsb)*lsb - lsb
    v_rp = (torch.max(array_in).item()//lsb)*lsb + lsb

    num_of_ramp_steps = int((v_rp-v_rn)/lsb + 1)

    ramp = torch.linspace(v_rn, v_rp, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1) #reshape
    # create array of comparator outputs
  
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next) #quantization
    # extract row and column tensors for all samples and ramp values

    num_rlevels = comp.size()[0]
    num_rows    = comp.size()[1]
    num_cols    = comp.size()[2]
    num_batches = comp.size()[3]

    row = comp.any(dim=2)
    col = comp.any(dim=1)

    #extract 3rd projection tensor
    #print(comp.cpu().numpy())
    digl = projection_onto_right_diagonal(comp.type(torch.uint8))
    digll = projection_onto_left_diagonal(comp.type(torch.uint8))


    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    #decoder starts here:
    # 0. Noise toss case: if all projections has a threshold value of wires active--> Noise range 6 rows cols and diags

    #These maps are used to select the batch and ramp using the comparison mask that is processible by a simple decoder
    ramp_idx_map  = torch.arange(0,num_rlevels, 1, dtype=torch.long, device=cuda).repeat(num_batches,1).T
    batch_idx_map =  torch.arange(0, num_batches, 1, dtype=torch.long, device=cuda).T.repeat(num_rlevels,1)
    cond_all_eq_two = torch.le(torch.sum(row, dim=1), 4) & torch.le(torch.sum(col, dim=1), 4) & torch.le(torch.sum(digl, dim=1), 4) & torch.le(torch.sum(digll, dim=1), 4)
    cond_any_le_one = torch.eq(torch.sum(row, dim=1), 1) | torch.eq(torch.sum(col, dim=1), 1) | torch.eq(torch.sum(digl, dim=1), 1) | torch.eq(torch.sum(digll, dim=1), 1) 

    cond_req1 = torch.eq(torch.sum(row, dim=1), 1)
    cond_ceq1 = torch.eq(torch.sum(col, dim=1), 1)
    cond_deq1 = torch.eq(torch.sum(digl, dim=1), 1)

    cond_req2 = torch.eq(torch.sum(row, dim=1), 2)
    cond_ceq2 = torch.eq(torch.sum(col, dim=1), 2)
    cond_deq2 = torch.eq(torch.sum(digl, dim=1), 2)

    oned_select_mask = cond_any_le_one | cond_all_eq_two
    zero_select_mask = torch.eq(torch.sum(row, dim=1), 0) & torch.eq(torch.sum(col, dim=1), 0) & torch.eq(torch.sum(digl, dim=1), 0) & torch.eq(torch.sum(digll, dim=1), 0)

    valid_ramp_indices = torch.masked_select(ramp_idx_map, oned_select_mask)
    valid_batch_indices = torch.masked_select(batch_idx_map, oned_select_mask)

    valid_col = col[valid_ramp_indices,:, valid_batch_indices]
    valid_row = row[valid_ramp_indices,:, valid_batch_indices]
    valid_dlg = digl[valid_ramp_indices,:, valid_batch_indices]
    valid_dlgl = digll[valid_ramp_indices,:, valid_batch_indices]

    bit_vld_col = (torch.sum(torch.eq(torch.sum(valid_col, dim=1), 1))*5 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 2))*10 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 3))*15 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 4))*20 + torch.sum(torch.eq(torch.sum(valid_col, dim=1), 5))*25 + torch.sum(torch.gt(torch.sum(valid_col, dim=1), 5))*32)
    bit_vld_dlg = (torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 2))*10 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 3))*15 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 4))*20 + torch.sum(torch.eq(torch.sum(valid_dlg, dim=1), 5))*25 + torch.sum(torch.gt(torch.sum(valid_dlg, dim=1), 5))*32) 
    bit_vld_dlgl = (torch.sum(torch.eq(torch.sum(valid_dlgl, dim=1), 3))*15 + torch.sum(torch.eq(torch.sum(valid_dlgl, dim=1), 4))*20 + torch.sum(torch.eq(torch.sum(valid_dlgl, dim=1), 5))*25 + torch.sum(torch.gt(torch.sum(valid_dlgl, dim=1), 5))*32) 
    bit_vld_row = (torch.sum(torch.eq(torch.sum(valid_row, dim=1), 1))*4 + torch.sum(torch.eq(torch.sum(valid_row, dim=1), 2))*8 + torch.sum(torch.eq(torch.sum(valid_row, dim=1), 3))*12+ torch.sum(torch.gt(torch.sum(valid_row, dim=1), 2))*16)


    bit_vld_tot = (bit_vld_col+bit_vld_row+bit_vld_dlg+bit_vld_dlgl).float()/(num_batches)

    print(bit_vld_tot)
    #print(compression_rate)

    xy_shadow = torch.einsum("bi,bj->bij",(valid_row.float(), valid_col.float()))

    diag_basis = projection_onto_right_basis((num_rows, num_cols))
    diagl_basis = projection_onto_left_basis((num_rows, num_cols))

    reordered_diag_basis = diag_basis.permute(1,0,2)
    reordered_diagl_basis =diagl_basis.permute(1,0,2)

    reordered_diag_shadow = torch.matmul(valid_dlg.float(), reordered_diag_basis.float())
    reordered_diagl_shadow = torch.matmul(valid_dlgl.float(), reordered_diagl_basis.float())

    diag_shadow = torch.einsum("ijk->jik", reordered_diag_shadow)
    diagl_shadow = torch.einsum("ijk->jik", reordered_diagl_shadow)

    valid_array_out = torch.einsum('bij,bij,bij->bij', (diag_shadow, xy_shadow, diagl_shadow))

    del diag_shadow, xy_shadow, reordered_diag_shadow, reordered_diag_basis, diag_basis
    del valid_col, valid_row, valid_dlg, ramp_idx_map, batch_idx_map, oned_select_mask
    torch.cuda.empty_cache()


    valid_naive_out = torch.zeros((num_rlevels, num_rows, num_cols, num_batches), device=cuda)
    valid_naive_out[valid_ramp_indices, :, :, valid_batch_indices] = valid_array_out

    reshape_naive_out = torch.einsum('hijk->kijh', valid_naive_out)
    naive_decoded = torch.matmul(reshape_naive_out, ramp.view(ramp.shape[0]).float())


    flat_naive_decoded = naive_decoded[:,idx_dim0, idx_dim1].view(samples_batch,array[0]*array[1])
    del naive_decoded, reshape_naive_out, valid_naive_out
    torch.cuda.empty_cache()

    return flat_naive_decoded



def rect_naive_interleaved_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
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
    rampup = rampup.view(rampup.shape[0])
    rampdown = rampdown.view(rampup.shape[0])
    #maskup = torch.zeros(1,1,array[0]*array[1], device=cuda, dtype=torch.int16)
    #maskup[0,0,lin_maskup] = 1
    #maskdown = torch.zeros(1, 1, array[0] * array[1], device=cuda, dtype=torch.int16)
    #maskdown[0,0,lin_maskdown] = 1
    # use torch.einsum to sum across all ramp values and get decoded output
    naive_decoded = torch.einsum("ijk,j->ik", (naive_valid*maskup_out.float(),rampup.float())) + \
                    torch.einsum("ijk,j->ik", (naive_valid*maskdown_out.float(),rampdown.float()))
    return naive_decoded

def rect_naive_tensor_profile(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    ramp = torch.linspace(v_rn, v_rp, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1)
    # create array of comparator outputs
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next)
    # extract row and column tensors for all samples and ramp values
    row = comp.any(dim=2)
    col = comp.any(dim=1)
    # free some space
    del array_in
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = torch.einsum("ij->ji", (valid,))
    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0], row.shape[1], row.shape[2], 1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    # free some space
    del row, col
    torch.cuda.empty_cache()
    # go back to all channels in one vector
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch, ramp.shape[0], array[0] * array[1])
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)
    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out, valid.float())
    # get data activity
    bit_tx_batch = torch.sum(naive_valid)/samples_batch
    return bit_tx_batch