import torch

class Einsum():
    def generate_einsum(self, global_params, local_params):
        def einsum(*input_tensors):
            ein_eq = local_params['equation']
            return [torch.einsum(ein_eq, *input_tensors)]
        return einsum

    def __call__(self, global_params, local_params):
        if not 'equation' in local_params:
            print('Missing Equation in Einsum Local Parameter List')
            exit()
        return self.generate_einsum(global_params,local_params)