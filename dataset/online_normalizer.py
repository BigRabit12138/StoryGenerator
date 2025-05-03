'''
在线获取标准化统计值
'''
import torch

class OnlineNormalizer:
    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n = n
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean)
        self.m2 = m2 if isinstance(m2, torch.Tensor) else torch.tensor(m2)

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        batch_size = x.numel()
        
        if batch_size == 0:
            return
            
        # 计算批次统计量
        batch_mean = torch.mean(x)
        batch_variance = torch.var(x, unbiased=True) if batch_size > 1 else torch.tensor(0.0)
        
        # 合并统计量
        if self.n == 0:
            self.mean = batch_mean
            self.m2 = batch_variance * (batch_size - 1) if batch_size > 1 else torch.tensor(0.0)
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_size
            self.mean = (self.n * self.mean + batch_size * batch_mean) / total_n
            self.m2 = (self.m2 + (batch_size - 1) * batch_variance + 
                      delta**2 * self.n * batch_size / total_n)
        
        self.n += batch_size

    def get_mean(self):
        return self.mean

    def get_std(self):
        if self.n < 2:
            return torch.tensor(0.0)
        return torch.sqrt(self.m2 / (self.n - 1))
    