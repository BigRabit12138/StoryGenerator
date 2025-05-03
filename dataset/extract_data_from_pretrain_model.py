'''
从预训练权重中提取训练数据
'''
import os
import json
import shutil
import signal
import threading

import torch
import asyncio

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from transformers import AutoConfig
from dataclasses import dataclass, field
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download

import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from dataset.online_normalizer import OnlineNormalizer


# ===== 配置常量 =====
DATA_PATH = Path("/root/autodl-tmp/PretrainData")  # 总数据存储根目录
MODEL_PATHS_FILE = Path(__file__).parent / "model_path.txt"  # 模型列表文件
STATUS_FILE = Path(__file__).parent / "status.json"  # 状态记录文件
SAFE_SHUTDOWN_FILE = Path(__file__).parent / "shutdown.lock"  # 安全关闭锁文件

# ===== 数据结构定义 =====
@dataclass
class ProcessingStatus:
    models: Dict[str, Dict] = field(default_factory=dict)  # 模型处理状态
    model_types: List[str] = field(default_factory=list)  # 模型类型列表
    is_download_finish: bool = field(default=False) # 下载进程是否结束
    max_dimension: int = field(default=0) # 所有矢量的最大维度
    norms: list[OnlineNormalizer] = field(default_factory=list) # 标准化统计值
    last_check_norms: list[dict] = field(default_factory=list)  # 上一次存储的状态

    @classmethod
    def from_dict(cls, data):
        norms = []
        for norm in data["norms"]:
            norms.append(
                OnlineNormalizer(
                    norm['n'],
                    norm['mean'],
                    norm['m2'],
                )
            )
        return cls(
            models=data["models"],
            model_types=data["model_types"],
            is_download_finish=data['is_download_finish'],
            max_dimension=data['max_dimension'],
            norms=norms,
            last_check_norms=data["norms"],
        )

    def to_dict(self, active=False):
        if active:
            norms = []
            for norm in self.norms:
                norms.append(
                    {
                        'n': norm.n,
                        'mean': norm.mean.item(),
                        'm2': norm.m2.item(),
                        'std': norm.get_std().item(),
                    }
                )
            self.last_check_norms = norms
        else:
            norms = status.last_check_norms
        return {
            "models": self.models,
            "model_types": self.model_types,
            "is_download_finish": self.is_download_finish,
            "max_dimension": self.max_dimension,
            "norms": norms,
        }

# ===== 全局状态和工具函数 =====
status = ProcessingStatus()
status_lock = threading.Lock()

def get_model_dir(model_name: str) -> Path:
    org, name = model_name.split("/")
    return DATA_PATH / org / name

async def download_model(model_name: str):
    model_dir = get_model_dir(model_name)
    try:
        await asyncio.to_thread(
            snapshot_download,
            repo_id=model_name,
            local_dir=model_dir,
        )

    except Exception as e:
        print(f"{model_name} 下载出现错误！\n{e}")
        return
    
    # 记录下载的权重文件
    weight_files = []
    for f in model_dir.glob("**/*"):
        if f.suffix in {".bin", ".pth", ".safetensors"}:
            weight_files.append(str(f.relative_to(model_dir)))
    
    status.models[model_name] = {
        "downloaded": True,
        "processed": False,
        "weight_files": weight_files,
        "processed_params": list(),
        "processed_files": list(),
    }
    save_status()  # 保存状态
    print(f"完成下载模型: {model_name}")


def process_weights(model_name: str, weight_file: str):
    model_dir = get_model_dir(model_name)
    file_path = model_dir / weight_file
    
    # 加载配置文件
    config = AutoConfig.from_pretrained(model_dir)
    with status_lock:
        if config.model_type not in status.model_types:
            status.model_types.append(config.model_type)
        
        model_idx = status.model_types.index(config.model_type)
    
    # 加载权重文件
    if file_path.suffix == ".safetensors":
        state_dict = load_file(file_path)
    else:
        state_dict = torch.load(file_path, map_location="cpu")
    
    # 创建输出目录
    org, name = model_name.split('/')
    output_dir = DATA_PATH / "TrainData" / org / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for param_name, tensor in tqdm(state_dict.items()):
        param_name = param_name.lower()
        with status_lock:
            if param_name in status.models[model_name]["processed_params"]:
                continue

        # 生成标签
        embedds = ['embed', ]
        layers = ['layer', ]
        heads = ['head', ]
        attns = ['attn', 'input',]
        mlps = ['mlp', 'post',]
        qs = ['q_', ]
        ks = ['k_', ]
        vs = ['v_', ]
        os = ['o_', ]
        lns = ['ln', 'layernorm']
        ups = ['up_', ]
        downs = ['down_', ]
        biases = ['bias', ]

        if any([True for embed in embedds if embed in param_name]):
            layer_type = 0
            sub_layer_idx = 0
            comp_type = 0
        elif any([True for layer in layers if layer in param_name]):
            layer_type = 1
            sub_layer_idx = int(param_name.split('.')[2]) + 1

            if any([True for attn in attns if attn in param_name]) \
                and any([True for ln in lns if ln in param_name]):
                comp_type = 1
            elif any([True for attn in attns if attn in param_name]) \
                and any([True for q in qs if q in param_name]):
                comp_type = 2
            elif any([True for attn in attns if attn in param_name]) \
                and any([True for k in ks if k in param_name]):
                comp_type = 3
            elif any([True for attn in attns if attn in param_name]) \
                and any([True for v in vs if v in param_name]):
                comp_type = 4
            elif any([True for attn in attns if attn in param_name]) \
                and any([True for o in os if o in param_name]):
                comp_type = 5
            
            elif any([True for mlp in mlps if mlp in param_name]) \
                and any([True for ln in lns if ln in param_name]):
                comp_type = 6
            elif any([True for mlp in mlps if mlp in param_name]) \
                and any([True for up in ups if up in param_name]):
                comp_type = 7
            elif any([True for mlp in mlps if mlp in param_name]) \
                and any([True for down in downs if down in param_name]):
                comp_type = 8

            else:
                comp_type = 9
        elif any([True for head in heads if head in param_name]):
            layer_type = 2
            sub_layer_idx = 0
            comp_type = 0
        else:
            layer_type = 3
            sub_layer_idx = 0
            comp_type = 0
        
        # 归一化
        layer_type  = layer_type / 3
        max_layer = 1 + max(
            [int(key.split('.')[2]) for key in state_dict.keys() if 'layer' in key]
        )
        sub_layer_idx = sub_layer_idx / max_layer
        comp_type = comp_type / 9

        def save_data(save_vector):
            label = torch.tensor([
                model_idx,
                layer_type,
                sub_layer_idx,
                comp_type,
                weight_type,
                weight_idx,
            ], dtype=torch.float32)
            
            # 保存数据
            data = {"X": save_vector, "Y": label}
            file_name = f"{model_idx}_{layer_type}_{sub_layer_idx}_{comp_type}_{weight_type}_{weight_idx}"
            save_file(data, str(output_dir / f"{file_name}.safetensors"))

        if any([True for bias in biases if bias in param_name]):
            weight_type = 0
            weight_idx = 0
            with status_lock:
                if tensor.shape[0] > status.max_dimension:
                    status.max_dimension = tensor.shape[0]
                for idx, one_item in enumerate(tensor):
                    if idx >= len(status.norms):
                        status.norms.append(OnlineNormalizer())
                    status.norms[idx].update(one_item)
            save_data(tensor)
        elif len(tensor.shape) == 1:
            weight_type = 1 / 2
            weight_idx = 0
            with status_lock:
                if tensor.shape[0] > status.max_dimension:
                    status.max_dimension = tensor.shape[0]
                for idx, one_item in enumerate(tensor):
                    if idx >= len(status.norms):
                        status.norms.append(OnlineNormalizer())
                    status.norms[idx].update(one_item)
            save_data(tensor)
        else:
            with status_lock:
                if tensor.shape[-1] > status.max_dimension:
                    status.max_dimension = tensor.shape[-1]
                for idx, one_item in enumerate(tensor.T):
                    if idx >= len(status.norms):
                        status.norms.append(OnlineNormalizer())
                    status.norms[idx].update(one_item)
            for weight_idx, vector in enumerate(tensor):
                weight_type = 2 / 2
                weight_idx = weight_idx / (len(tensor) - 1)
                save_data(vector)
                
        # 更新处理状态
        with status_lock:
            status.models[model_name]["processed_params"].append(param_name)
            save_status(active=True)  # 保存状态
    # 更新处理状态
    with status_lock:
        status.models[model_name]["processed_files"].append(weight_file)
        save_status(active=True)  # 保存状态


async def download_worker():
    """负责下载模型的异步任务"""
    with open(MODEL_PATHS_FILE) as f:
        models = [line.strip() for line in f if line.strip()]
    
    for model_name in models:
        with open(MODEL_PATHS_FILE) as f:
            for line in f:
                if line.strip() not in models:
                    models.append(line.strip())
        if model_name in status.models and status.models[model_name]["downloaded"]:
            print(f"{model_name} 已经下载！")
            continue
        
        print(f"开始下载模型: {model_name}")
        await download_model(model_name)
    status.is_download_finish = True

async def process_worker(poll_interval=5):
    """持续检查 status.models 中的模型，处理已下载但未处理的模型"""
    while True:
        # 找出已下载但未处理的模型
        pending_models = [
            model_name for model_name, info in status.models.items()
            if info.get("downloaded") and not info.get("processed")
        ]

        is_finish = [info.get("processed") for info in status.models.values()]

        if status.is_download_finish and all(is_finish):
            print("✅ 所有模型均已处理，退出任务。")
            break  # 所有模型都处理完了，结束任务

        for model_name in pending_models:
            model_dir = get_model_dir(model_name)
            print(f"开始处理模型: {model_name}")
            
            # 并行处理未处理的权重文件
            tasks = [
                asyncio.to_thread(process_weights, model_name, wf)
                for wf in status.models[model_name]["weight_files"]
                if wf not in status.models[model_name]["processed_files"]
            ]
            await asyncio.gather(*tasks)

            # 更新状态
            status.models[model_name]["processed"] = True
            save_status(active=True)  # 保存状态

            # 清理模型目录
            try:
                shutil.rmtree(model_dir)
                print(f"已清理模型目录: {model_dir}")
            except Exception as e:
                print(f"清理目录失败: {str(e)}")
            
            print(f"✅ 完成处理模型: {model_name}")

        # 等待一会再检查是否有新模型添加
        await asyncio.sleep(poll_interval)


async def main_processing():
    """主处理函数，并行运行下载和处理任务"""
    # 创建任务
    download_task = asyncio.create_task(download_worker())
    process_task = asyncio.create_task(process_worker())
    
    # 等待两个任务完成
    await asyncio.gather(download_task, process_task)


# ===== 信号处理和状态保存 =====
def save_status(active=False):
    with open(STATUS_FILE, "w") as f:
        json.dump(status.to_dict(active), f)

def signal_handler(sig, frame):
    print("\n捕获中断信号，保存状态...")
    save_status()
    if Path(SAFE_SHUTDOWN_FILE).exists():
        Path(SAFE_SHUTDOWN_FILE).unlink()
    os._exit(0)

if __name__ == "__main__":
    # 初始化信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 加载已有状态
    if Path(STATUS_FILE).exists():
        with open(STATUS_FILE) as f:
            status = ProcessingStatus.from_dict(json.load(f))
    
    # 创建安全关闭锁
    Path(SAFE_SHUTDOWN_FILE).touch()
    
    try:
        asyncio.run(main_processing())
    except Exception as e:
        print(f"处理出错: {str(e)}")
        save_status()
    finally:
        if Path(SAFE_SHUTDOWN_FILE).exists():
            Path(SAFE_SHUTDOWN_FILE).unlink()