from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter as scatter
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import get_laplacian, remove_self_loops
from torch_sparse import SparseTensor, matmul


class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension))

    def reset_parameters(self):
        pass

    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output


class SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_m = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=bias)

    def reset_parameters(self):
        self.lin_r.reset_parameters()

        self.lin_m.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_t: Tensor,
    ) -> Tensor:

        row, col = edge_index
        x_j = torch.cat([x[col], edge_attr, edge_t], dim=1)
        x_j = scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce="sum")
        x_j = self.lin_m(x_j)
        x_i = self.lin_r(x)
        out = 0.5 * x_j + x_i

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

