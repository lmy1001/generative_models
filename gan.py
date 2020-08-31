import numpy as np
import torch
import torch.nn as nn
from generators_discriminators import mlp_discriminator, point_cloud_generator
import torch.autograd as autograd

class w_gan_gp(nn.Module):
    def __init__(self, generator, discriminator,
                 noise_dim=128, npoints=2048, batch_size=50, lam=10):
        super(w_gan_gp, self).__init__()
        self.noise_dim = noise_dim
        self.npoints = npoints
        self.batch_size = batch_size

        self.generator = generator
        self.discriminator = discriminator
        self.lam = lam

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand((self.batch_size, 1, 1), dtype=torch.float32)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.cuda() if torch.cuda.is_available() else alpha
        # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = (alpha * real_samples) + ((1 - alpha) * fake_samples)
        interpolates = interpolates.cuda() if torch.cuda.is_available() else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        interpolates_logit, _ = self.discriminator(interpolates)

        #fake = autograd.Variable((real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = torch.ones(interpolates_logit.size()).cuda() if torch.cuda.is_available() else torch.ones(
            interpolates_logit.size())
        gradients = autograd.grad(
            outputs=interpolates_logit,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, x, z):
        x_transpose = torch.transpose(x, 1, 2)

        real_logit, real_prob = self.discriminator(x_transpose)
        gen_out = self.generator(z)
        gen_out_transpose = torch.transpose(gen_out, 1, 2)

        synthetic_logit, synthetic_prob = self.discriminator(gen_out_transpose)
        loss_d = torch.mean(synthetic_logit) - torch.mean(real_logit)
        loss_g = -torch.mean(synthetic_logit)

        #compute gradient penality
        gradient_penalty = self.compute_gradient_penalty(x_transpose, gen_out_transpose)
        loss_gradient_penalty = self.lam * gradient_penalty
        loss_d += loss_gradient_penalty

        return gen_out, loss_d, loss_g

class vanilla_gan(nn.Module):
    def __init__(self, generator, discriminator,
                 noise_dim=128, npoints=2048, batch_size=50):
        super(vanilla_gan, self).__init__()
        self.noise_dim = noise_dim
        self.npoints = npoints
        self.batch_size = batch_size
        self.discriminator = discriminator
        self.generator = generator

    def forward(self, x, z):
        x_transpose = torch.transpose(x, 1, 2)
        real_logit, real_prob = self.discriminator(x_transpose)
        gen_out = self.generator(z)
        gen_out_transpose = torch.transpose(gen_out, 1, 2)
        synthetic_logit, synthetic_prob = self.discriminator(gen_out_transpose)

        loss_d = torch.mean(-torch.log(real_prob) - torch.log(1 - synthetic_prob))
        loss_g = torch.mean(-torch.log(synthetic_prob))

        return gen_out, loss_d, loss_g


if __name__=='__main__':
    noise_dim = 128
    input = torch.randn(5, 10, 3)
    mu, sigma=0, 0.2
    batch_size = 5
    npoints=10
    net = w_gan_gp(noise_dim, npoints, batch_size)
    z = torch.Tensor(np.random.normal(mu, sigma, (batch_size, noise_dim)))
    lam = 0.5
    gen_out, loss_d, loss_g, loss_gradient_penalty = net(input, z, lam)
    print(gen_out.shape)
    print(loss_d)
    print(loss_g)
    print(loss_gradient_penalty)



