import argparse
import os
import random
import sys
import time

import dgl
import dgl.function as fn


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    MultiLayerNeighborSampler,
)
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn


def load_proteins(root, dataset = 'ogbn-proteins'):
    n_node_feats, n_edge_feats, n_classes = 0, 8, 112
    data = DglNodePropPredDataset(name=dataset, root = root)
    # evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    graph.ndata["labels"] = labels


    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(
        fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat")
    )
    n_node_feats = graph.ndata["feat"].shape[-1]

    # Only the labels in the training set are used as features, while others are filled with zeros.
    graph.ndata["train_labels_onehot"] = torch.zeros(
        graph.num_nodes(), n_classes
    )
    graph.ndata["train_labels_onehot"][train_idx, labels[train_idx, 0]] = 1
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    graph.create_formats_()

    # return graph, labels

    return data, graph, labels, train_idx, val_idx, test_idx #, evaluator


# def preprocess(graph, labels, train_idx):
#     global n_node_feats

