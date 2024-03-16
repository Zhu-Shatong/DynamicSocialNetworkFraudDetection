import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter as scatter
from torch import Tensor
import numpy as np


def add_degree_feature(x: Tensor, edge_index: Tensor):
    """_summary_ 向特征矩阵中添加节点的入度和出度作为新的特征

    Args:
        x (Tensor):  输入特征矩阵
        edge_index (Tensor): 边的索引
 
    Returns:
        _type_:  添加了入度和出度特征的特征矩阵
    """
    # edge_index 是图中所有边的索引
    row, col = edge_index
    # 计算入度，即每个节点作为边的终点的次数
    in_degree = torch_geometric.utils.degree(col, x.size(0), x.dtype)

    # 计算出度，即每个节点作为边的起点的次数
    out_degree = torch_geometric.utils.degree(row, x.size(0), x.dtype)
    # 将原始特征与计算出的入度和出度特征拼接起来
    return torch.cat([x, in_degree.view(-1, 1), out_degree.view(-1, 1)], dim=1)


def add_feature_flag(x):
    """_summary_ : 为特征矩阵添加特征标记
    对特征矩阵中的特定值（-1）标记，并将这些值替换为0。

    Args:
        x (_type_):  输入特征矩阵

    Returns:
        _type_: 添加了特征标记的特征矩阵
    """
    # 创建一个与x相同形状的零矩阵，用于标记特征
    feature_flag = torch.zeros_like(x[:, :17])
    # 将原始特征矩阵中等于-1的元素对应的标记矩阵位置设置为1
    feature_flag[x[:, :17] == -1] = 1
    # 将原始特征矩阵中等于-1的元素替换为0
    x[x == -1] = 0
    # 将原始特征矩阵和标记矩阵拼接起来
    return torch.cat((x, feature_flag), dim=1)


def add_label_feature(x, y):
    """_summary_ : 为特征矩阵添加标签特征
    将标签（y）转换为独热编码，并添加到特征矩阵中。

    Args:
        x (_type_):  输入特征矩阵
        y (_type_):  标签

    Returns:
        _type_:  添加了标签特征的特征矩阵
    """
    y = y.clone()
    # 将标签y中的1（表示欺诈节点）暂时替换为0，模拟从正常用户中挖掘欺诈用户的场景
    y[y == 1] = 0

    print(y)

    # 对标签进行独热编码，并且去除最后一个特征（为了避免多重共线性）
    y_one_hot = F.one_hot(y).squeeze()
    # 将原始特征和独热编码后的标签拼接起来
    return torch.cat((x, y_one_hot[:, :-1]), dim=1)


def add_label_counts(x, edge_index, y):
    """_summary_ : 为特征矩阵添加标签计数特征
    计算每个节点邻居的标签统计，并将其添加到特征矩阵中。

    Args:
        x (_type_):  输入特征矩阵
        edge_index (_type_):   边的索引
        y (_type_):  标签

    Returns:
        _type_:  添加了标签计数特征的特征矩阵
    """
    y = y.clone().squeeze()
    # 确定背景节点和前景节点
    background_nodes = torch.logical_or(y == 2, y == 3)
    foreground_nodes = torch.logical_and(y != 2, y != 3)
    y[background_nodes] = 1
    y[foreground_nodes] = 0

    row, col = edge_index
    # 对边缘节点的标签进行独热编码
    a = F.one_hot(y[col])
    b = F.one_hot(y[row])
    # 计算每个节点的邻居标签统计信息
    temp = scatter.scatter(a, row, dim=0, dim_size=y.size(0), reduce="sum")
    temp += scatter.scatter(b, col, dim=0, dim_size=y.size(0), reduce="sum")

    # 将原始特征和邻居标签统计信息拼接起来
    return torch.cat([x, temp.to(x)], dim=1)


def cos_sim_sum(x, edge_index):
    """_summary_ : 计算并添加余弦相似度特征
    计算每个节点与其邻居之间的余弦相似度之和，并将其添加到特征矩阵中。
    
    Args:
        x (_type_): 输入特征矩阵
        edge_index (_type_): 边的索引

    Returns:
        _type_: 添加了余弦相似度特征的特征矩阵
    """
    row, col = edge_index
    # 计算边缘节点特征之间的余弦相似度
    sim = F.cosine_similarity(x[row], x[col])
    # 对每个节点的邻居的相似度进行求和
    sim_sum = scatter.scatter(
        sim, row, dim=0, dim_size=x.size(0), reduce="sum")
    # 将原始特征和相似度求和结果拼接起来
    return torch.cat([x, torch.unsqueeze(sim_sum, dim=1)], dim=1)


def to_undirected(edge_index, edge_attr):
    """_summary_ : 将有向边转换为无向边

    Args:
        edge_index (_type_):   边的索引
        edge_attr (_type_): 边的属性

    Returns:
        _type_:  无向边的索引和属性
    """
    # 将有向边转换为无向边
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    # 将边的属性也进行相应的复制
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    return edge_index, edge_attr


def data_process(data):
    # 对数据进行预处理
    edge_index, edge_attr = (
        data.edge_index,
        data.edge_attr,
    )

    x = data.x
    # 为特征矩阵添加度特征
    x = add_degree_feature(x, edge_index)
    # 计算并添加余弦相似度特征
    x = cos_sim_sum(x, edge_index)
    # 将边转换为无向边
    edge_index, edge_attr = to_undirected(
        edge_index, edge_attr
    )
    # 过滤掉重复的边
    mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    # 再次确保边是无向的
    data.edge_index, data.edge_attr = to_undirected(
        edge_index, edge_attr
    )

    # 设置边的方向标记
    data.edge_direct = torch.ones(data.edge_attr.size(0), dtype=torch.long)
    data.edge_direct[: data.edge_attr.size(0) // 2] = 0

    # 添加特征标记
    x = add_feature_flag(x)
    # 添加标签计数特征
    x = add_label_counts(x, edge_index, data.y)
    # 添加标签特征（这一步被注释掉了）
    # x = add_label_feature(x, data.y)
    data.x = x
    # 如果标签维度是2，则将其压缩为1维
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    return data
