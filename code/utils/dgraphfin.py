from typing import Optional, Callable, List
import os.path as osp

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


def degree_feat(edge_index, x):
    """`_summary_` : 为特征矩阵添加节点的入度和出度作为新的特征

    Args: 
        edge_index (_type_): 边的索引
        x (_type_):     输入特征矩阵

    Returns:
        _type_:  添加了入度和出度特征的特征矩阵
    """
    adj = csr_matrix(
        (np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])),
        shape=(x.shape[0], x.shape[0]),
    )
    out_degree, in_degree = adj.sum(axis=1), adj.sum(axis=0).T
    return out_degree, in_degree


def edge_type_feat(edge_type, edge_index, x):
    """_summary_ : 为特征矩阵添加边的类型作为新的特征

    Args:
        edge_type (_type_): 边的类型
        edge_index (_type_):  边的索引
        x (_type_):    输入特征矩阵

    Returns:
        _type_: 添加了边的类型特征的特征矩阵
    """
    edge_type_adj = csr_matrix(
        (edge_type, (edge_index[:, 0], edge_index[:, 1])),
        shape=(x.shape[0], x.shape[0]),
    )
    edge_type_feat = np.zeros((x.shape[0], 11))
    data, indptr = edge_type_adj.data, edge_type_adj.indptr
    for i in range(x.shape[0]):
        row = data[indptr[i]: indptr[i + 1]]
        unique, counts = np.unique(row, return_counts=True)
        for j, k in zip(unique, counts):
            edge_type_feat[i, j - 1] = k
    return edge_type_feat


def read_dgraphfin(folder):
    print("read_dgraphfin")
    names = ["dgraphfin.npz"]
    items = [np.load(folder + "/" + name) for name in names]

    x = items[0]["x"]
    y = items[0]["y"].reshape(-1, 1)

    edge_index = items[0]["edge_index"]
    edge_type = items[0]["edge_type"]

    np.random.seed(42)

    train_mask_t = items[0]['train_mask']
    np.random.shuffle(train_mask_t)
    train_mask = train_mask_t[:int(len(train_mask_t)/10*7)]
    valid_mask = train_mask_t[int(len(train_mask_t)/10*7):]
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index.transpose(),
                              dtype=torch.long).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    train_mask = torch.tensor(train_mask, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.valid_mask = valid_mask

    return data


class DGraphFin(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"xygraphp1"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ""

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):

        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["dgraphfin.npz"]
        return names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data = read_dgraphfin(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
