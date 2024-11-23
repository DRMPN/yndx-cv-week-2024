import torch
from torch.distributions import Independent, Normal, Bernoulli

d, nh, D = 32, 200, 28 * 28


def loss_vae(x, encoder, decoder):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    mu_logvar = encoder(x)
    mu, logvar = torch.chunk(mu_logvar, 2, dim=1)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std

    x_recon_logits = decoder(z)

    recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        x_recon_logits, x, reduction='none'
    )
    recon_loss = recon_loss.view(x.size(0), -1).sum(dim=1)

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    loss = recon_loss + kl_divergence

    loss = loss.mean()

    return loss, x_recon_logits
  
