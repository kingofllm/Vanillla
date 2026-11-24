import os
import io
import glob
import torch
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Sequence
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import random
import cv2  # 引入 cv2 进行加速
import math

# 限制 OpenCV 线程数，防止与 PyTorch DataLoader 冲突
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class StreamingSupervisedDataset(IterableDataset):
    def __init__(self, parquet_files, dataset_type="cifar", transform=None, use_coarse=False):
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.dataset_type = dataset_type
        self.transform = transform
        self.use_coarse = use_coarse

        # 预先计算总长度，用于 tqdm 显示进度条
        self.total_len = 0
        print(f"[INFO] Scanning {len(parquet_files)} files to calculate total length...")
        for f in parquet_files:
            # 仅读取元数据，不加载数据，速度很快
            self.total_len += pq.read_metadata(f).num_rows
        print(f"[INFO] Total samples: {self.total_len}")

    def __len__(self):
        return self.total_len

    def _process_image(self, img_bytes):
        """使用 cv2 高速解码图像"""
        # 1. 转换为 numpy 数组
        np_arr = np.frombuffer(img_bytes, np.uint8)
        # 2. 解码 (BGR)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # 3. 容错处理：如果解码失败
        if img is None:
            return None
        # 4. 转为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __iter__(self):
        """
        核心流式加载逻辑
        """
        worker_info = get_worker_info()

        # 1. 确定当前 Worker 负责哪些文件
        if worker_info is None:
            # 单进程模式
            my_files = self.parquet_files
        else:
            # 多进程模式：将文件列表均匀分配给每个 Worker
            my_files = [
                f for i, f in enumerate(self.parquet_files)
                if i % worker_info.num_workers == worker_info.id
            ]

        # 2. 【文件级顺序】：不再打乱文件列表 my_files。Worker 将按文件名的自然顺序处理它们。

        # 3. 逐个加载文件
        for file_path in my_files:
            # print(f"[Worker {worker_info.id if worker_info else 0}] Loading: {os.path.basename(file_path)}")

            try:
                # 读取整个 parquet 文件到内存
                df = pq.read_table(file_path).to_pandas()

                # 4. 【文件内打乱】：打乱当前 DataFrame 的行
                df = df.sample(frac=1.0).reset_index(drop=True)

                # 5. 迭代 DataFrame 中的每一行
                for _, row in df.iterrows():

                    # 获取图片字节
                    if "img" in row:
                        img_bytes = row["img"]["bytes"]
                    elif "image" in row:
                        img_bytes = row["image"]["bytes"]
                    else:
                        continue  # 跳过无效数据

                    # 解码图片
                    img_array = self._process_image(img_bytes)
                    if img_array is None:
                        continue

                    # 转为 PIL 以适配 torchvision transforms
                    from PIL import Image
                    img = Image.fromarray(img_array)

                    # 数据增强
                    if self.transform:
                        img = self.transform(img)

                    # 获取标签
                    result = {"image": img, "idx": -1}

                    if self.dataset_type == "cifar":
                        result["fine_idx"] = int(row.get("fine_label", -1))
                        result["coarse_idx"] = int(row.get("coarse_label", -1))
                    elif self.dataset_type in ["stanford-cars", "tiny-imagenet", "nabirds"]:
                        result["labels"] = int(row.get("label", -1))

                    yield result

                # 处理完一个文件后，显式删除 df，帮助 GC 回收内存
                del df

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue


@dataclass
class DataCollatorForSupervisedDataset:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 过滤掉 None (解码失败的样本)
        instances = [i for i in instances if i is not None]
        if len(instances) == 0:
            return None

        images = torch.stack([inst["image"] for inst in instances])
        keys = instances[0].keys()
        batch = {"images": images}
        if "fine_idx" in keys:
            batch["fine_idx"] = torch.tensor([inst["fine_idx"] for inst in instances])
        if "coarse_idx" in keys:
            batch["coarse_idx"] = torch.tensor([inst["coarse_idx"] for inst in instances])
        if "labels" in keys:
            batch["labels"] = torch.tensor([inst["labels"] for inst in instances])

        return batch


# --- 数据增强定义保持不变 ---
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_stanford_cars_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_stanford_cars_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_tiny_imagenet_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_tiny_imagenet_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_nabirds_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_nabirds_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def make_supervised_data_module(data_args) -> Dict:
    dataset_type = data_args.dataset_type.lower()

    # Worker 初始化函数
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if dataset_type == "cifar":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))

        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type="cifar",
            transform=transform_cifar_train,
            use_coarse=data_args.use_coarse
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type="cifar",
            transform=transform_cifar_test,
            use_coarse=data_args.use_coarse
        )

    elif dataset_type == "stanford-cars":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))

        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_stanford_cars_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_stanford_cars_test
        )
    elif dataset_type == "tiny-imagenet":

        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))

        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_tiny_imagenet_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_tiny_imagenet_test
        )

    elif dataset_type == "nabirds":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))
        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_nabirds_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_nabirds_test
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    data_collator = DataCollatorForSupervisedDataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}