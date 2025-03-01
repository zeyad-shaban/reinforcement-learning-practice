import torch
from torch.utils.tensorboard import SummaryWriter


def write_grad_info(parameters: list[torch.Tensor], writer: SummaryWriter, category: str, x, is_grad=False):
    if is_grad:
        grads = torch.cat([p.flatten() for p in parameters if p is not None])
    else:
        grads = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
    max_grad = grads.max()
    l2_grad = grads.norm()
    var_grad = grads.var()

    writer.add_scalar(f'{category}/grad/max', max_grad, x)
    writer.add_scalar(f'{category}/grad/l2', l2_grad, x)
    writer.add_scalar(f'{category}/grad/var', var_grad, x)
