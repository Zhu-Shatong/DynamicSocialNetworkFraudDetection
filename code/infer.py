import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.subgraph import k_hop_subgraph

from feat_func import data_process
from models import GEARSage
from utils import DGraphFin
from utils.evaluator import Evaluator
from utils.utils import prepare_folder


def set_seed(seed):
    np.random.seed(seed)             # 设置NumPy的随机种子
    random.seed(seed)                # 设置Python内置random模块的随机种子
    torch.manual_seed(seed)          # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)     # 设置PyTorch在CUDA上的随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机种子


@torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
def test(model, data):
    model.eval()  # 将模型设置为评估模式

    # 传递整个图通过模型
    out = model(
        data.x, data.edge_index, data.edge_attr, data.edge_direct,
    )

    y_pred = out.exp()  # 计算预测值的指数，用于将输出转换为概率
    return y_pred


def main():

    # 创建一个解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(
        description="GEARSage for DGraphFin Dataset")

    # 添加不同的命令行参数
    parser.add_argument("--dataset", type=str, default="DGraphFin")
    parser.add_argument("--model", type=str, default="GEARSage")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hiddens", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # 解析命令行参数
    args = parser.parse_args()
    print("args:", args)

    # 判断是否有可用的CUDA设备，如果有，使用CUDA设备；否则使用CPU
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("device:", device)

    # 设置随机种子，以确保实验可重复
    set_seed(42)

    # 加载DGraphFin数据集
    dataset = DGraphFin(root="./dataset", name="DGraphFin")
    nlabels = 2  # 设置标签数量
    data = dataset[0]  # 获取数据

    # 展示一下大小
    print('data.test_mask.shape:', data.test_mask.shape)
    print('data.train_mask.shape:', data.train_mask.shape)
    print('data.valid_mask.shape:', data.valid_mask.shape)
    print('data.y.shape:', data.y.shape)

    # 划分数据集为训练集、验证集和测试集
    split_idx = {
        "train": data.train_mask,
        "valid": data.valid_mask,
        "test": data.test_mask,
    }

    # 对数据进行预处理并转移到指定设备
    data = data_process(data).to(device)

    # 获取训练集索引，并根据标签分为正负样本
    train_idx = split_idx["train"].to(device)

    data.train_pos = train_idx[data.y[train_idx] == 1]
    data.train_neg = train_idx[data.y[train_idx] == 0]

    # 初始化模型
    model = GEARSage(
        in_channels=data.x.size(-1),
        hidden_channels=args.hiddens,
        out_channels=nlabels,
        num_layers=args.layers,
        dropout=args.dropout,
        activation="elu",
        bn=True,
    ).to(device)

    print(f"Model {args.model} initialized")

    model_file = './model_files/{}/{}/model.pt'.format(
        args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file))

    out = test(model, data)

    evaluator = Evaluator('auc')
    preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    train_auc = evaluator.eval(y_train, preds_train)['auc']
    valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
    print('train_auc:', train_auc)
    print('valid_auc:', valid_auc)

    preds = out[data.test_mask].cpu().numpy()

    np.save('./submit/preds.npy', preds)


if __name__ == "__main__":
    main()
