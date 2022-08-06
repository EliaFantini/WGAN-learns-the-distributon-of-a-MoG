import torch

from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=100):
        super().__init__()
        self.linear1 = nn.Linear(noise_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        z = nn.functional.relu(self.linear1(z))
        z = self.linear2(z)
        return z


class DualVariable(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, c=1e-2):
        super().__init__()
        self.c = c
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def enforce_lipschitz(self, spectral_norm=True):
        """Enforce the 1-Lipschitz condition of the function by doing weight clipping or spectral normalization"""
        if spectral_norm:
            self.spectral_normalisation()
        else:
            self.weight_clipping()

    def spectral_normalisation(self, use_approximation=True):
        """
        Perform spectral normalisation, forcing the singular value of the weights to be upper bounded by 1.
        """
        with torch.no_grad():
            for module in [self.linear1, self.linear2]:
                if use_approximation:
                    h = module.weight.data.shape[0]
                    u = torch.Tensor(module.weight.data.new(h).normal_(0, 1))
                    v = torch.Tensor(module.weight.data.new(h).normal_(0, 1))
                    W = torch.Tensor(module.weight.data)
                    v.data = torch.mv(torch.t(W.view(h, -1).data), u.data) / torch.mv(torch.t(W.view(h, -1).data),
                                                                                      u.data).norm(p=2)
                    u.data = torch.mv(W.view(h, -1).data, v.data) / torch.mv(W.view(h, -1).data, v.data).norm(p=2)
                    sigma_w = torch.t(u).dot(W.view(h, -1).mv(v)).T
                    module.weight.data = module.weight.data / sigma_w
                else:
                    module.weight.data = module.weight.data / module.weight.data.svd().S.max()

    def weight_clipping(self):
        """
        Clip the parameters to $-c,c$. You can access a modules parameters via self.parameters().
        Remember to access the parameters  in-place and outside of the autograd with Tensor.data.
        """
        with torch.no_grad():
            for p in self.parameters():
                p.data = torch.clamp(p.data, - self.c, self.c)
