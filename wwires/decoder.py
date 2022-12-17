import torch

class SimpleDecoder():
    def generate_simple_decoder(self, global_params, local_params):
        def simple_decoder(input_tensor):
            print(f'Simple Decoder: Parameters = {global_params}, {local_params}')
            return [torch.einsum('ij->ji', input_tensor)]

        return simple_decoder

    def __call__(self, global_params, local_params={}):
        return self.generate_simple_decoder(global_params, local_params)