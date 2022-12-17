import torch

class SimpleEncoder():
    def generate_simple_encoder(self, global_params, local_params):
        def simple_encoder(input_tensor):
            print(f'Simple Encoder: Parameters = {global_params}, {local_params}')
            return [torch.einsum('ij->ji', input_tensor)]

        return simple_encoder

    def __call__(self, global_params, local_params={}):
        return self.generate_simple_encoder(global_params, local_params)

class ProjectionEncoder():
    def generate_projection_encoder(self, global_params, local_params):
        direction = local_params['direction']
        def x_proj_encoder(data):
            return [data.any(dim=2)]

        def y_proj_encoder(data):
            return [data.any(dim=1)]

        def dl_proj_encoder(data):
            num_rlevels = global_params['num_rlevels']
            N = global_params['num_rows']
            M = global_params['num_cols']
            num_batches = global_params['num_batches']

            assert(N <= M)

            diag_or = torch.zeros([num_rlevels, M, num_batches], dtype=torch.uint8).cuda()
            boundary = M - N + 1

            for ii in range(M):
                if ii < boundary:
                    diag_or[:,ii,:] = torch.diagonal(data, offset=ii, dim1=-2, dim2=-1).any(dim=2)
                else:
                    diag_or[:,ii,:] = torch.cat(
                                            (torch.diagonal(data, offset=ii, dim1=-2, dim2=-1),
                                             torch.diagonal(data, offset=ii-M, dim1=-2, dim2=-1)),
                                            -1).any(dim=2)
            return [diag_or]


        proj_fxn = {'x' : x_proj_encoder, 'y' : y_proj_encoder, 'dl' : dl_proj_encoder}

        return proj_fxn[direction]

    def __call__(self, global_params, local_params={'direction':'x'}):
        return self.generate_projection_encoder(global_params, local_params)
