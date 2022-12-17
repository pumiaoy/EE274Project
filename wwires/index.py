import torch

class BatchIndexMap:
    def generate_batch_index_map(self, global_params, local_params):
        def batch_index_map():
            row = global_params[local_params['row']]
            col = global_params[local_params['col']]

            row_idx_map = torch.arange(0, row, 1, dtype=torch.long).cuda().repeat(col,1).T
            col_idx_map = torch.arange(0, col, 1, dtype=torch.long).cuda().T.repeat(row,1)

            return [row_idx_map, col_idx_map]
        return batch_index_map

    def __call__(self, global_params, local_params):
        return self.generate_batch_index_map(global_params, local_params)