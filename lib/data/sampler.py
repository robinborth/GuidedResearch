from torch.utils.data import Sampler


class SimpleIndexSampler(Sampler):
    def __init__(self, idxs: list[int]):
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
