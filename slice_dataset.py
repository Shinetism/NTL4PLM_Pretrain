import numpy as np
from fairseq.data import data_utils, BaseWrapperDataset

class SliceDataset(BaseWrapperDataset):
    """Create a dataset slice"""

    def __init__(self, dataset, length):
        super().__init__(dataset)
        assert length is not None
        assert length <= len(self.dataset)
        self.length = length
        self.dataset = dataset

    def __getitem__(self, index):
        if index >= self.length:
            raise ValueError
        return self.dataset[index]

    @property
    def sizes(self):
        return self.dataset.sizes[:self.length]
    
    def __len__(self):
        return self.length
