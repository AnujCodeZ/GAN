import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake, device='cpu'):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated = real * epsilon + fake * (1 - epsilon)
    
    mixed_scores = critic(interpolated)
    
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    
    return torch.mean((gradient_norm - 1) ** 2)
    