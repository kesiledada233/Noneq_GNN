import torch
import torch.optim as optim


r'''
SGD每次使用mini-batch训练时只使用一部分训练数据，计算的梯度本身就具有随机性。
这意味着SGD本身就是一种随机优化算法，也就是说SGD本质上已经像Langevin方程一样包含了一个噪声项。
但是SGD的随机性主要来自于数据采样，而不是参数更新。
最重要的是！SGD里的噪声大小是隐含的！无法直接进行控制。
因此，我们在SGD的基础上显式地加入一个噪声项，使得噪声的大小可以被控制，这样就能主动调控系统的非平衡行为。
'''


class LangevinOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, beta=0.1, anneal_factor=0.99):
        defaults = dict(lr=lr, beta=beta, anneal_factor=anneal_factor)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            anneal_factor = group['anneal_factor']

            for p in group['params']: # 遍历模型参数
                if p.grad is None: # 跳过没有梯度的参数
                    continue
                d_p = p.grad.data # 计算参数的梯度
                noise = torch.randn_like(d_p) * beta  # 加入高斯噪声
                p.data.add_(-lr, d_p + noise)  # Langevin 下降法 (θ = θ - α∇L(θ) + βη)
                group['beta'] *= anneal_factor  # 逐步降低噪声

        return loss