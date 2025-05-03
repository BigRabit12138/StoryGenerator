import json
import torch

from pathlib import Path
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence

class PretrainWeightDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        """
        参数:
            mode (str): 必须是 'train', 'val' 或 'test' 之一
            dataset_path (str): 数据集目录路径
        """
        assert mode in ['train', 'val', 'test'], "mode 必须是 'train', 'val' 或 'test'"
        self.mode = mode
        
        # 加载文件路径列表
        with open(Path(__file__).parent / f'{mode}.txt', 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        
        # 加载标准化统计信息
        with open(Path(__file__).parent / 'status.json', 'r') as f:
            stats = json.load(f)
            self.norms = stats['norms']
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 从safetensors文件加载数据
        data = load_file(file_path)
        X = data['X']  # 原始特征向量
        Y = data['Y']  # 标签值
        
        # 假设 self.norms['mean'] 和 self.norms['std'] 已经是张量形式
        mean = torch.tensor(
            [norm['mean'] for idx, norm in enumerate(self.norms) if idx < len(X)],
            device=X.device
        )
        std = torch.tensor(
            [norm['std'] for idx, norm in enumerate(self.norms) if idx < len(X)],
            device=X.device
        )

        # 一次性标准化全部数据
        normalized_X = (X - mean) / std
        
        # 生成位置序列
        input_ids = torch.arange(len(X))
        
        # 返回所需数据
        return {
            'locations': Y,
            'input_ids': input_ids,
            'labels': normalized_X
        }

    @staticmethod
    def collate_fn(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list]:
        locations = torch.stack([item['locations'] for item in batches])
        length = [len(item['input_ids']) for item in batches]
        max_len = max(length)
        idx = length.index(max_len)
        input_ids = torch.stack([batches[idx]["input_ids"] for _ in batches])
        labels = [item['labels'] for item in batches]
        padding_labels = pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100.0,
        )

        # 3. 构造 mask（1=真实标签，0=填充部分）
        masks = torch.ones(len(labels), max_len)  # 初始化全 1
        for i, item in enumerate(labels):
            masks[i, len(item):] = 0  # 将填充位置设为 0
        return {
            'locations': locations,
            'input_ids': input_ids,
            'labels': padding_labels,
            'masks': masks,
        }