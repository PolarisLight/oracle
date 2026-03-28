import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm  # 导入 tqdm 库
import concurrent.futures  # 导入多进程库

def filter_classes_by_sample_count(data_dir, min_samples=2):
    """
    过滤掉样本少于 min_samples 的类别
    :param data_dir: 数据集根目录
    :param min_samples: 最少样本数，排除少于该数量的类别
    :return: 筛选后的有效类别路径
    """
    class_samples = defaultdict(list)
    
    # 遍历文件夹中的每个类别
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        # 确保是文件夹
        if os.path.isdir(class_path):
            # 获取该类文件夹下的所有文件
            class_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # 如果该类别文件数大于等于 min_samples，则保留
            if len(class_files) >= min_samples:
                class_samples[class_name] = class_files
    
    return class_samples

def split_dataset(class_samples, train_ratio=0.7, random_seed=42):
    """
    按照指定比例划分训练集和测试集
    :param class_samples: 筛选后的类别样本
    :param train_ratio: 训练集比例
    :param random_seed: 随机种子
    :return: 训练集和测试集文件路径
    """
    random.seed(random_seed)

    train_files = []
    test_files = []

    # 遍历每个类别，并使用 tqdm 显示进度条
    for class_name, files in tqdm(class_samples.items(), desc="Splitting dataset", unit="class"):
        # 打乱文件顺序
        random.shuffle(files)
        
        # 按照比例划分训练集和测试集
        train_size = int(len(files) * train_ratio)
        
        train_files.extend([(os.path.join(class_name, file), 'train') for file in files[:train_size]])
        test_files.extend([(os.path.join(class_name, file), 'test') for file in files[train_size:]])

    return train_files, test_files

def copy_file(src, dst):
    """
    复制文件
    :param src: 源文件路径
    :param dst: 目标文件路径
    """
    shutil.copy(src, dst)

def save_to_folder(files, data_dir, output_dir, split):
    """
    将文件保存到指定的文件夹中
    :param files: 包含文件路径的列表
    :param data_dir: 数据集根目录
    :param output_dir: 输出目录
    :param split: 'train' 或 'test'（用于创建不同的目录结构）
    """
    # 使用 tqdm 显示处理进度
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for file, split in tqdm(files, desc=f"Saving {split} set", unit="file"):
            class_name = os.path.dirname(file)
            class_folder = os.path.join(output_dir, split, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            # 构建源文件路径
            src_path = os.path.join(data_dir, class_name, os.path.basename(file))
            dst_path = os.path.join(class_folder, os.path.basename(file))
            
            # 异步提交文件复制任务
            futures.append(executor.submit(copy_file, src_path, dst_path))
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)

def create_data_split(data_dir, output_dir, train_ratio=0.7, min_samples=2):
    """
    根据需求创建数据集划分：排除少样本类别，并划分训练集与测试集
    :param data_dir: 数据集根目录
    :param output_dir: 输出目录，用于保存训练集和测试集
    :param train_ratio: 训练集比例
    :param min_samples: 排除的最小样本数
    """
    # 步骤 1: 过滤掉只有1个样本的类
    class_samples = filter_classes_by_sample_count(data_dir, min_samples)
    
    # 步骤 2: 按照比例划分训练集和测试集
    train_files, test_files = split_dataset(class_samples, train_ratio)
    
    # 步骤 3: 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # 步骤 4: 保存训练集和测试集
    save_to_folder(train_files, data_dir, output_dir, 'train')
    save_to_folder(test_files, data_dir, output_dir, 'test')



data_dir = 'H:\DatasetD\OBC306'  # 数据集根目录
output_dir = 'H:\DatasetD\OBC306_total'  # 输出目录
create_data_split(data_dir, output_dir, train_ratio=1, min_samples=2)
