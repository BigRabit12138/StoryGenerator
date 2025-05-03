'''
划分训练测试集
'''
import os
import glob
import random

from pathlib import Path
    

def find_files_recursive(directory, extension):
    """
    递归查找目录下所有指定扩展名的文件
    
    :param directory: 要搜索的根目录
    :param extension: 文件扩展名 (如 '*.safetensors')
    :return: 匹配文件的完整路径列表
    """
    search_pattern = os.path.join(directory, '**', extension)
    return glob.glob(search_pattern, recursive=True)


def main():
    base_dir = "/root/autodl-tmp/PretrainData/TrainData"
    
    files = find_files_recursive(base_dir, '*.safetensors')
    if not files:
        print("No .safetensors files found")
    print(f"Found {len(files)} files")

    random.seed(42)
    random.shuffle(files)
    train_data = files[:int(len(files)/2)]
    val_data = files[int(len(files)/2):int(len(files)/4*3)]
    test_data = files[int(len(files)/4*3):]

    with open(Path(__file__).parent / 'train.txt', "w") as f:
        f.writelines('\n'.join(train_data))
    with open(Path(__file__).parent / 'val.txt', "w") as f:
        f.writelines('\n'.join(val_data))
    with open(Path(__file__).parent / 'test.txt', "w") as f:
        f.writelines('\n'.join(test_data))

    print("Done.")

if __name__ == "__main__":

    main()
