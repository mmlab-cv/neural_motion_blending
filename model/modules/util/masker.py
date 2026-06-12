import torch
import torch.nn as nn

from einops import rearrange

class Conv():
    """
    Wrapper for convolution operations.
    """
    def _conv_nd(dims, *args, **kwargs):
        """ Create a 1D, 2D, or 3D convolution module. """
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    def _zero_module(module):
        """ Zero out the parameters of a module and return it. """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def zero_conv(dims, *args, **kwargs):
        """ Create a zero-initialized convolutional layer. """
        return Conv._zero_module(Conv._conv_nd(dims, *args, **kwargs))


class SkeletonMasker(nn.Module):
    """ Wrapper to build stratified masks for skeletal data. """
    
    def __init__(self, c_prob=0.0, t_prob=0.0, j_prob=0.0, min_chains=1):
        super().__init__()
        self.c_mask_prob = c_prob
        self.t_mask_prob = t_prob
        self.j_mask_prob = j_prob
        self.min_chains = min_chains

    def forward(self, control):
        if self.training:
            return self._train_step(control.x, control.y['kinematic_chains'])
        else:
            return self._inference_step(control.x, control.prompts)

    def _train_step(self, x, kinematic_chains):
        """ Stratified sparse random masking """
        bs, _, njoints, _, nframes = x.shape
        # Build component masks
        if self.c_mask_prob > 0.0:
            c_mask = self._chain(x, kinematic_chains).view(bs, 1, njoints, 1, 1)
            x = x * c_mask
        if self.t_mask_prob > 0.0:
            t_mask = self._temporal(x).view(1, 1, 1, 1, nframes)
            x = x * t_mask
        if self.j_mask_prob > 0.0:
            j_mask = self._joint(x).view(1, 1, njoints, 1, 1)
            x = x * j_mask
        return x
    
    def _inference_step(self, x, prompts):
        """ Mask joints prompt """
        return x         
        bs, c, j, _, _ = x.shape
        j_mask = torch.zeros((bs * c, j), device=x.device)
        for i, p in enumerate(prompts):
            j_mask[i, p[0]] = 1.0
        return x * j_mask.view(bs, c, j, 1, 1)
    
    @staticmethod
    def _bernoulli(size, prob, device):
        if prob <= 0.0:
            return torch.ones(size, device=device)
        return (torch.rand(size, device=device) >= prob).float()

    def _chain(self, x, kinematic_chains):
        if self.c_mask_prob <= 0.0:
            return torch.ones((x.shape[0], x.shape[2]), device=x.device)
        bs, _, njoints, _, _ = x.shape
        mask = torch.zeros((bs, njoints), device=x.device)
        for b in range(bs):
            chains = kinematic_chains[b]
            n_c = len(chains)
            n_keep = max(self.min_chains, min(int(n_c * (1 - self.c_mask_prob)), n_c - 1))
            perm = torch.randperm(n_c, device=x.device)[:n_keep]
            for i in perm:
                mask[b, chains[i]] = 1.0
        return mask

    def _joint(self, x):
        return self._bernoulli(x.shape[2], self.j_mask_prob, x.device)

    def _temporal(self, x):
        return self._bernoulli(x.shape[-1], self.t_mask_prob, x.device)