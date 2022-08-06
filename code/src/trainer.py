import uuid
import tempfile
import os
import torch
import math

from tqdm import tqdm
from . import plots

temp = tempfile.gettempdir()


class GanTrainer():
    """
    Trains a WGAN with different discriminator regularization strategies

    Args:
        batch_size (int): batch size
        data (callable): real data distribution.
        noise (callable): noise distribution.
        mode (str): whether to use "penalty" or to "clip" the dual
        make_gif (bool): Whether to save a 2D snapshot of the samples per
            iteration and compose an animated gif.
        checkpoints (int): number of images to save as pdf
    """
    def __init__(
            self, batch_size, data, noise, make_gif=False):
        self.data = data
        self.noise = noise
        self.id = uuid.uuid4()
        self.snapshots = []
        self.checkpoints = []
        self.batch_size = batch_size
        self.make_gif = make_gif
        self.fixed_real_sample = data.dist.sample((1000,)).cpu().numpy()
        self.step=0

    def _snapshot(self, g, ckpt=False):
        """Save an image of the current generated samples"""
        with torch.no_grad():
            gen_sample = g(self.noise.sample((self.batch_size,))).cpu().numpy()
        file_png = os.path.join(
                temp, str(self.id) + '_' + str(len(self.snapshots)) + '.png')
        filename = [file_png]
        if ckpt:
            file_pdf = os.path.join(
                    str(self.id) + '_' + str(len(self.checkpoints)) + '.pdf')
            filename.append(file_pdf)
        plots.compare_samples_2D(
                self.fixed_real_sample, gen_sample, filename)
        self.snapshots.append(filename[0])
        if ckpt:
            self.checkpoints.append(filename[1])

    def render_gif(self, output_file, duration):
        """
        Render animated gif based on current snapshots

        Args:
            output_file (str): output_file
            duration (float): output video duration in seconds
        """
        plots.animate(self.snapshots, output_file, duration)

    @staticmethod
    def objective(f, g, data_sample, noise_sample):
        """
        Minimax objective of the GANs

        Args:
            data_sample (torch.tensor): sample from the true distribution
            noise_sample (torch.tensor): sample from the noise distribution

        Remember that this should be E[f(x) - f(g(z))], where E is the expectation, x is the data,
        z is the noise and f and g are dual variable and generator respectively.

        Calculate an estimate of the expectation via the empirical mean on a sample.
        """
        W1 = (f(data_sample) - f(g(noise_sample))).mean()
        return W1

    def alternating_update(self, f, g, f_optim, g_optim,f_ratio=1):
        """
        Update dual variable, then update generator.
        Refer to the tutorial/pytorch documentation 

        for the dual variable, remember to enforce the lipschitz constraint
        after each update. Also recall that by default pytorch optimizers
        minimize an objective, whereas the optimization problem of the
        dual variable is a 'maximization' problem.
        """
        noise = self.noise.sample([self.batch_size])
        real = self.data.sample([self.batch_size])
        if self.step % f_ratio == 0:
            f.eval()
            g.train()

            g_optim.zero_grad()
            objective_g = self.objective(f, g, real, noise)
            objective_g.backward()
            g_optim.step()
        else:
            f.train()
            g.eval()

            f_optim.zero_grad()
            objective_f = -self.objective(f, g, real, noise)
            objective_f.backward()
            f_optim.step()
            f.enforce_lipschitz()

    def alternating(self, n_iter, f, g, f_optim, g_optim, n_checkpoints, f_ratio=1):
        """
        Update generator and discriminator a number of iterations via
        alternating gradient descent/ascent.

        Args:
            n_iter (int):
            f (nn.Module):
            g (nn.Module):
            f_optim (optim.Optimizer):
            g_optim (optim.Optimizer):
            n_checkpoints (int):
        """
        ckpts = math.floor(n_iter / n_checkpoints)
        bar=tqdm(range(n_iter))
        for _ in bar:
            self.step += 1
            l=self.alternating_update(f, g, f_optim, g_optim,f_ratio=f_ratio)
            if self.make_gif:
                if _ % ckpts == 0 or n_iter - _ == 10: # the added condition is to save a pdf also on last iterations ( 10 iterations before stopping )
                    self._snapshot(g, ckpt=True)
                else:
                    self._snapshot(g, ckpt=False)

