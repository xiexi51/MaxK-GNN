import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math

class MaxK(Function):
    @staticmethod
    def forward(ctx, input, k=1):
        topk, indices = input.topk(k, dim=1)
        mask = torch.zeros_like(input)
        mask.scatter_(1, indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        # Multi-layers SAGEConv
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
                # norm_layer = nn.BatchNorm1d(hid_size)
            else:
                norm_layer = None
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean", feat_drop=feat_drop, norm=norm_layer))
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean", feat_drop=feat_drop))

        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

        self.nonlinear = nonlinear
    def forward(self, g, x):
        x = self.lin_in(x)

        for i in range(self.num_layers):
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            # x = self.dropout(x)
            x = self.layers[i](g, x)
        x = self.lin_out(x)

        return x
    

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)



        self.nonlinear = nonlinear
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        x = self.lin_out(x)
        return x

class GIN(nn.Module):
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.pytorch.conv.GINConv(learn_eps=True, activation=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)



        self.nonlinear = nonlinear
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        x = self.lin_out(x)
        return x
    
class GNN_res(nn.Module):
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers1 = nn.ModuleList()
        self.dropoutlayers2 = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers1.append(nn.Dropout(feat_drop))
            self.dropoutlayers2.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                # self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        self.reslayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers1.append(Linear(hid_size, hid_size))
            self.linlayers2.append(Linear(hid_size, hid_size))
            self.reslayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers1[i].weight)
            init.xavier_uniform_(self.linlayers2[i].weight)
            init.xavier_uniform_(self.reslayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x_res = self.reslayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)

            x = self.linlayers1[i](x)
            x = F.relu(x)
            x = self.dropoutlayers1[i](x)
            x = self.linlayers2[i](x)
            
            x = x_res + x
            x = F.relu(x)
            x = self.dropoutlayers2[i](x)

        x = self.lin_out(x)
        return x