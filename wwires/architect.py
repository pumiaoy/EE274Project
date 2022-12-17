from wwires import BlockBuilder, Configuration, DeleteTensors
import torch

class GraphArchitect:
    def __init__(self, system_name, path_name='config/system'):
        self.system_name = system_name
        self.system_config = Configuration(config_name=system_name, path_head=path_name)
        self.block_builder = BlockBuilder()
        self.system_graph, self.system_params = self.load_graph()

    def load_graph(self):
        execution_graph = {}

        block_list = self.system_config['blocks']
        available_blocks = self.block_builder.available_blocks()
        build = self.block_builder

        for (ii, block_info) in enumerate(block_list):
            block = list(block_info.keys())[0]
            block_obj = build[block]()
            if block in available_blocks:
                execution_graph[ii] = (block_obj, block_info[block])
            else:
                print(f'{block} is not a known block type, add block into the Block Builder Table')

        return execution_graph, self.system_config['parameters']

    def execute(self, input_tensors):
        intermediate_outputs = {}

        graph = self.system_graph
        global_params = self.system_params

        node_obj, node_properties = graph[0]
        inp_tensors = input_tensors # {inp : inp_tnsr for (inp, inp_tnsr) in zip(node_properties['inputs'], input_tensors)}

        params = [global_params]
        if 'parameters' in node_properties:
            local_params = node_properties['parameters']
            params += [local_params]

        if 'output' in node_properties:
            out_tensor_keys = node_properties['output']
            out_tensors = node_obj(*params)(*inp_tensors)

            for (out_tensor_key, out_tensor) in zip(out_tensor_keys, out_tensors):
                intermediate_outputs[out_tensor_key] = out_tensor
                del out_tensor
            del out_tensors
            torch.cuda.empty_cache()
        else:
            out_tensor_keys = None
            node_obj(*params)(*inp_tensors)


        for ii in range(1, len(graph)):
            node_obj, node_properties = graph[ii] 

            #Handle Unique Memory Freeing Operation
            if type(node_obj) is DeleteTensors:
                for inp in node_properties['inputs']:
                    del intermediate_outputs[inp]
                torch.cuda.empty_cache()
                continue 
            #inp_tensors = {inp : intermediate_outputs[inp] for inp in node_properties['inputs']}
            if 'inputs' in node_properties:
                print(node_properties['inputs'])
                inp_tensors = [intermediate_outputs[inp] for inp in node_properties['inputs']]
            else:
                inp_tensors = []

            params = [global_params]
            if 'parameters' in node_properties:
                local_params = node_properties['parameters']
                params += [local_params]

            if 'output' in node_properties:
                out_tensor_keys = node_properties['output']
                out_tensors = node_obj(*params)(*inp_tensors)

                for (out_tensor_key, out_tensor) in zip(out_tensor_keys, out_tensors):
                    intermediate_outputs[out_tensor_key] = out_tensor
                    print(out_tensor)
                    del out_tensor
                del out_tensors
                torch.cuda.empty_cache()
            else:
                out_tensor_keys = None
                node_obj(*params)(*inp_tensors)

        self.intermediate_outputs = intermediate_outputs

        if out_tensor_keys is None:
            return None

        return [intermediate_outputs[out_tensor_key] for out_tensor_key in out_tensor_keys]

    def __getitem__(self, name):
        return self.system_config[name]

    def __setitem__(self, name, value):
        self.system_config[name] = value
