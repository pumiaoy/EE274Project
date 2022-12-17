from wwires import (
                        SimpleDecoder, 
                        SimpleEncoder, 
                        ProjectionEncoder, 
                        Interpolater,
                        OffsetCorrecter,
                        Einsum,
                        RampQuantizer,
                        DeleteTensors,
                        BatchIndexMap,
                        SummationBound,
                        LogicalReduction)

class BlockBuilder:
    def __getitem__(self, block_type):
        return self.generator_table[block_type]

    def available_blocks(self):
        return list(self.generator_table.keys())

    generator_table = {'SimpleEncoder':SimpleEncoder,
                       'ProjectionEncoder':ProjectionEncoder,
                       'SimpleDecoder':SimpleDecoder,
                       'OffsetCorrecter': OffsetCorrecter,
                       'Interpolater':Interpolater,
                       'Einsum':Einsum,
                       'RampQuantizer':RampQuantizer,
                       'DeleteTensors':DeleteTensors,
                       'BatchIndexMap':BatchIndexMap,
                       'SummationBound':SummationBound,
                       'LogicalReduction':LogicalReduction}