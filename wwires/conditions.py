import torch

class SummationBound:
    def generate_sum_bound(self, global_params, local_params):
        bound_table = {'le' : torch.le, 'ge' : torch.ge, 'eq' : torch.eq, 'lt': torch.lt, 'gt':torch.gt}
        def sum_bound(input_tensor):
            dim = local_params['dimension']
            bound_type, bound_value = local_params['bound'].split()
            return [bound_table[bound_type](torch.sum(input_tensor, dim=dim), float(bound_value))]
        return sum_bound
    def __call__(self, global_params, local_params):
        return self.generate_sum_bound(global_params, local_params)

class LogicalReduction:
    def generate_logic_reduce(self, global_params, local_params):
        def logic_and_reduce(*input_tensor):
            lgc_rdc_tensor = input_tensor[0]
            print(input_tensor)
            for inp_ten in input_tensor[1:]:
                lgc_rdc_tensor = lgc_rdc_tensor & inp_ten
            print(lgc_rdc_tensor)
            return [lgc_rdc_tensor]

        def logic_or_reduce(*input_tensor):
            lgc_rdc_tensor = input_tensor[0]
            print(input_tensor)
            for inp_ten in input_tensor[1:]:
                lgc_rdc_tensor = lgc_rdc_tensor | inp_ten
            print(lgc_rdc_tensor)
            return [lgc_rdc_tensor]


        def logic_xor_reduce(*input_tensor):
            lgc_rdc_tensor = input_tensor[0]
            for inp_ten in input_tensor[1:]:
                lgc_rdc_tensor = lgc_rdc_tensor ^ inp_ten
            return [lgc_rdc_tensor]

        reduce_oper = local_params['reduce_with']

        reduce_fxn = { 'or': logic_or_reduce, 'and': logic_and_reduce, 'xor': logic_xor_reduce }

        return reduce_fxn[reduce_oper]

    def __call__(self,global_params,local_params):
        return self.generate_logic_reduce(global_params, local_params)