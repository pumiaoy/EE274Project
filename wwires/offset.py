import torch

class OffsetCorrecter:
    def generate_offset_correction(self, global_params, local_params):
        def offset_correction(data):
            return [(data - torch.mean(data, 1, True)).type(torch.int16)]
        return offset_correction

    def __call__(self, global_params, local_params={}):
        return self.generate_offset_correction(global_params, local_params)