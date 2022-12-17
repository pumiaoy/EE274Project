import torch

class RampQuantizer:
    def generate_ramp_quantizer(self, global_params, local_params):
        def ramp_quantizer(array_in):
            lsb = global_params['lsb']
            v_rn = global_params['v_rn']
            v_rp = global_params['v_rp']
            num_rlevels = global_params['num_rlevels']

            ramp = torch.linspace(v_rn, v_rp, num_rlevels).type(torch.int16).cuda().view(num_rlevels, 1, 1, 1)
            return [torch.ge(array_in, ramp) & torch.lt(array_in, ramp + lsb)]
        return ramp_quantizer
        
    def __call__(self, global_params, local_params={}):
        return self.generate_ramp_quantizer(global_params, local_params)