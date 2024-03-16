import torch
import torch.nn.functional as F
import torch.nn as nn


# 导入自定义模块中的组件
from models.layers import (
    TimeEncoder,  # 时间编码器
    SAGEConv,     # SAGE卷积层
)

# 创建激活函数层


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()  # 没有激活函数，返回恒等变换
    elif activation == "relu":
        return nn.ReLU()      # ReLU激活函数
    elif activation == "elu":
        return nn.ELU()       # ELU激活函数
    else:
        raise ValueError("Unknown activation")  # 未知激活函数类型


# 定义GEARSage模型类，继承自nn.Module
class GEARSage(nn.Module):
    def __init__(
        self,
        in_channels,        # 输入特征维度
        hidden_channels,    # 隐藏层特征维度
        out_channels,       # 输出特征维度
        edge_attr_channels=50,  # 边属性维度
        time_channels=50,       # 时间特征维度
        num_layers=2,           # 网络层数
        dropout=0.0,            # Dropout比率
        bn=True,                # 是否使用批标准化
        activation="elu",       # 激活函数类型
    ):

        super().__init__()
        self.convs = nn.ModuleList()  # 卷积层列表
        self.bns = nn.ModuleList()    # 批标准化层列表
        bn = nn.BatchNorm1d if bn else nn.Identity

        # 构建网络层
        for i in range(num_layers):  # 遍历网络层
            # 第一层的输入特征维度为输入特征维度，其余层为隐藏层特征维度
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - \
                1 else hidden_channels  # 最后一层的输出特征维度为输出特征维度，其余层为隐藏层特征维度
            self.convs.append(  # 添加卷积层
                SAGEConv(
                    (
                        first_channels + edge_attr_channels + time_channels,  # 输入特征维度
                        first_channels,                                  # 输出特征维度
                    ),
                    second_channels,                                  # 输出特征维度
                )
            )
            self.bns.append(bn(second_channels))  # 添加批标准化层

        self.dropout = nn.Dropout(dropout)       # Dropout层
        self.activation = creat_activation_layer(activation)  # 激活函数层
        self.emb_type = nn.Embedding(12, edge_attr_channels)  # 边类型嵌入
        self.emb_direction = nn.Embedding(2, edge_attr_channels)  # 边方向嵌入
        self.t_enc = TimeEncoder(time_channels)  # 时间编码器
        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        # 参数初始化

        for conv in self.convs:  # 遍历卷积层
            conv.reset_parameters()  # 卷积层参数初始化

        for bn in self.bns:  # 遍历批标准化层
            if not isinstance(bn, nn.Identity):  # 如果不是恒等变换
                bn.reset_parameters()  # 批标准化层参数初始化

        nn.init.xavier_uniform_(self.emb_type.weight)  # 边类型嵌入参数初始化

        nn.init.xavier_uniform_(self.emb_direction.weight)  # 边方向嵌入参数初始化

    def forward(self, x, edge_index, edge_attr, edge_t):
        # 前向传播
        edge_attr = self.emb_type(edge_attr)  # 边属性嵌入
        edge_t = self.t_enc(edge_t)  # 时间特征编码
        for i, conv in enumerate(self.convs):  # 遍历卷积层
            x = conv(x, edge_index, edge_attr, edge_t)  # 卷积层
            x = self.bns[i](x)  # 批标准化
            x = self.activation(x)  # 激活函数
            x = self.dropout(x)  # Dropout

        return x.log_softmax(dim=-1)  # 返回对数Softmax输出
