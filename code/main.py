# 训练模型

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
    """_summary_ : 设置随机种子，以确保实验可重复

    Args:
        seed (_type_):  随机种子
    """
    np.random.seed(seed)             # 设置NumPy的随机种子
    random.seed(seed)                # 设置Python内置random模块的随机种子
    torch.manual_seed(seed)          # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)     # 设置PyTorch在CUDA上的随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机种子


def train(model, data, optimizer):
    """_summary_ : 训练模型

    Args:
        model (_type_):  模型
        data (_type_):  数据
        optimizer (_type_):  优化器

    Returns:
        _type_: 损失值
    """
    model.train()  # 将模型设置为训练模式

    optimizer.zero_grad()  # 清除之前的梯度

    # 从负样本中随机选取与正样本数量相同的样本，以保持平衡
    neg_idx = data.train_neg[
        torch.randperm(data.train_neg.size(0))[: data.train_pos.size(0)]
    ]
    train_idx = torch.cat([data.train_pos, neg_idx], dim=0)

    # 提取k-hop子图
    nodeandneighbor, edge_index, node_map, mask = k_hop_subgraph(
        train_idx, 3, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0)
    )

    # 通过模型传递子图
    out = model(
        data.x[nodeandneighbor],
        edge_index,
        data.edge_attr[mask],
        data.edge_direct[mask],
    )

    # 计算损失，使用负对数似然损失函数
    loss = F.nll_loss(out[node_map], data.y[train_idx])
    loss.backward()  # 反向传播计算梯度

    nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # 梯度裁剪，防止梯度爆炸

    optimizer.step()  # 更新模型参数
    torch.cuda.empty_cache()  # 清空CUDA缓存，节省内存
    return loss.item()  # 返回损失值


@torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
def test(model, data):
    """_summary_: 评估模型

    Args:
        model (_type_): 模型
        data (_type_): 数据

    Returns:
        _type_:  预测值
    """
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
    parser.add_argument("--dataset", type=str, default="DGraphFin")  # 数据集名称
    parser.add_argument("--model", type=str, default="GEARSage")  # 模型名称
    parser.add_argument("--device", type=int, default=0)  # 设备
    parser.add_argument("--epochs", type=int, default=500)  # 训练轮数
    parser.add_argument("--hiddens", type=int, default=96)  # 隐藏层维度
    parser.add_argument("--layers", type=int, default=2)  # 层数
    parser.add_argument("--dropout", type=float, default=0.3)  # Dropout概率

    # 解析命令行参数
    args = parser.parse_args()
    print("args:", args)

    # 判断是否有可用的CUDA设备，如果有，使用CUDA设备；否则使用CPU
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("device:", device)

    # 准备模型文件夹
    model_dir = prepare_folder(args.dataset, args.model)
    print("model_dir:", model_dir)

    # 设置随机种子
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

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    # 设置评估器
    best_auc = 0.0
    evaluator = Evaluator("auc")

    # 获取训练集和验证集的标签
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]

    # 训练模型
    for epoch in range(1, args.epochs + 1):

        loss = train(model, data, optimizer)
        out = test(model, data)
        preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
        train_auc = evaluator.eval(y_train, preds_train)["auc"]
        valid_auc = evaluator.eval(y_valid, preds_valid)["auc"]

        # 保存表现最好的模型
        if valid_auc >= best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), model_dir + "model.pt")
            preds = out[data.test_mask].cpu().numpy()
        print(
            f"Epoch: {epoch:02d}, "
            f"Loss: {loss:.4f}, "
            f"Train: {train_auc:.2%}, "
            f"Valid: {valid_auc:.2%},"
            f"Best: {best_auc:.4%},"
        )

    # 在测试集上评估模型
    test_auc = evaluator.eval(data.y[data.test_mask], preds)["auc"]
    print(f"test_auc: {test_auc}")


if __name__ == "__main__":
    main()
