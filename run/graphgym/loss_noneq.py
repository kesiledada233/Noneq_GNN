import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgym.register as register
from graphgym.config_noneq import cfg


def compute_energy(node_feature, edge_index, alpha=0.5, beta=2.0):
    """
    Compute energy of GNN

    Args:
        node_feature (torch.tensor): Node feature matrix

    Returns: Energy of GNN
        A项: 节点嵌入L2范数平方
        B项: 邻接节点之间嵌入差异的L2范数平方

    """
    # Compute energy of GNN

    # A项: 节点嵌入L2范数平方和
    # 对node_feature的每一行计算L2范数，node_feature的形状为[num_nodes, num_features]，则经过torch.norm()后输出的形状为[num_nodes], 表示每个节点特征的L2范数
    # 最后再对所有节点的L2范数平方求平均值，得到一个标量energy_a, 表示所有节点嵌入的L2范数平方和的平均值
    energy_a = torch.mean(torch.norm(node_feature, dim=1) ** 2)

    # B项: 领接节点之间嵌入差异的L2范数平方和
    # edge_index的形状为[2, num_edges], 取出row和col分别表示边的起始节点和终止节点
    # 对node_feature的row和col索引的节点特征进行相减，得到每一条边的嵌入差异
    # 然后对每一条边的嵌入差异计算L2范数，得到一个形状为[num_edges]的张量
    # 最后对所有边的L2范数平方求平均值，得到一个标量energy_b, 表示所有边的嵌入差异的L2范数平方和的平均值
    row, col = edge_index
    diff = node_feature[row] - node_feature[col]
    energy_b = torch.mean(torch.norm(diff, dim=1) ** 2)

    # 计算GNN能量
    return alpha * energy_a + beta * energy_b


def compute_loss(pred, true, batch=None, h=None):
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    ret = 0

    if cfg.model.loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            ret = 1
            pred = F.log_softmax(pred, dim=-1)
            loss_pred = F.nll_loss(pred, true)
        else:
            ret = 2
            true = true.float()
            loss_pred = bce_loss(pred, true)
    elif cfg.model.loss_fun == 'mse':
        ret = 3
        true = true.float()
        loss_pred = mse_loss(pred, true)
    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

    # 加入能量正则项
    if cfg.train.get('energy_reg', False) and batch is not None:
        energy_loss = compute_energy(
            h,
            batch.edge_index,
            alpha=cfg.train.energy_alpha,
            beta=cfg.train.energy_beta,
        )
        loss = loss_pred + energy_loss
    else:
        loss = loss_pred

    if ret == 1:
        return loss, pred
    elif ret == 2:
        return loss, torch.sigmoid(pred)
    elif ret == 3:
        return loss, pred
