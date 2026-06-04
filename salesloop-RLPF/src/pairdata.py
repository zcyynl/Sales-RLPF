import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import random

class BalancedBatchSampler(Sampler):
    """
    保证每个 batch 至少有 1 个正样本和 1 个负样本
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # 存正负样本 index
        self.pos_indices = [i for i, y in enumerate(dataset.label) if y == 1]
        self.neg_indices = [i for i, y in enumerate(dataset.label) if y == 0]

    def __iter__(self):
        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        while len(neg_pool) >= (self.batch_size - 1) and len(pos_pool) >= 1:
            # 从正样本池选 1 条
            pos = [pos_pool.pop()]
            # 从负样本池选 (batch_size - 1) 条
            neg = [neg_pool.pop() for _ in range(self.batch_size - 1)]
            batch = pos + neg
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(self.pos_indices), len(self.neg_indices)) // self.batch_size



class BalancedBatchSampler_new(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_indices = [i for i, y in enumerate(dataset.label) if y == 1]
        self.neg_indices = [i for i, y in enumerate(dataset.label) if y == 0]

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        # 按 GPU rank 切分数据（每卡取一部分）
        pos_pool = pos_pool[self.rank::self.world_size]
        neg_pool = neg_pool[self.rank::self.world_size]

        while len(pos_pool) > 0 and len(neg_pool) >= (self.batch_size - 1):
            batch = [pos_pool.pop()]  # 每个 batch 至少一个正样本
            batch += [neg_pool.pop() for _ in range(self.batch_size - 1)]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.world_size)