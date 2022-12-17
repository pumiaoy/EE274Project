import torch

class Interpolater:
    def generate_interpolater(self, global_params, local_params):
        def interpolater(decoded_data):
            num_batches = local_params['num_batches']
            num_values  = global_params['num_channels']

            sigma = local_params['sigma']
            cuda = torch.device('cuda')

            interp_data  = torch.zeros((num_batches, num_values), device=cuda)
            lgc_mat_zero = torch.eq(decoded_data[1:num_batches-1,:],0).type(torch.FloatTensor).to(cuda)
            
            interp_left  = decoded_data[0:num_batches-2,:]
            interp_right = decoded_data[1:num_batches-1,:]

            interp_points = lgc_mat_zero*(torch.randn_like(lgc_mat_zero)*sigma + (interp_left + interp_right)/2)

            interp_data[1:num_batches-1,:]  = decoded_data[1:num_batches-1,:] + interp_points

            return [interp_data]
        return interpolater

    def __call__(self, global_params, local_params={'sigma' : 2, 'num_batches' : 128}):
        return self.generate_interpolater(global_params, local_params)