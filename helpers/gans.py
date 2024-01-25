import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from helpers.Config import Config
from helpers.Discriminator import Discriminator
from helpers.Generator import Generator
from helpers.utils import print_real_fake_images


def get_disc_loss(gen, disc, criterion, real, num_images, noise_dimension, device):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # All of them will got label as 0
    # .detach() here is to ensure that only discriminator parameters will get update
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred,
                               torch.zeros_like(disc_fake_pred))

    # Pass real features to discriminator
    # All of them will got label as 1
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred,
                               torch.ones_like(disc_real_pred))

    # Average of loss from both real and fake features
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, noise_dimension, device):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # But all of them will got label as 1
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


def tensor_to_images(image_tensor, num_images=30, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=6)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_vector_dimension, device='cpu'):
    return torch.randn(n_samples, noise_vector_dimension,device=device)


def train(config: Config, dataloader: DataLoader, gen: Generator, gen_opt: torch.optim, disc: Discriminator, disc_opt: torch.optim):
    mean_generator_loss, mean_discriminator_loss = 0, 0
    overall_mean_generator_loss, overall_mean_discriminator_loss = [], []

    n = len(dataloader)
    print("DataLoader length: ", n)

    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")

        mean_generator_loss, mean_discriminator_loss = 0, 0
        for i, data in enumerate(tqdm(dataloader)):
            real, _ = data
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(config.device)

            # Training discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, config.criterion, real, cur_batch_size, config.noise_dimension,
                                      config.device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Training generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, config.criterion, cur_batch_size, config.noise_dimension, config.device)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / config.display_step
            mean_generator_loss += gen_loss.item() / config.display_step

        if epoch % config.display_step == 0:
            # Generate noise and pass to generator
            fake_noise = get_noise(cur_batch_size, config.noise_dimension, device=config.device)
            fake = gen(fake_noise)

            print_real_fake_images(config, real, fake, epoch)

        overall_mean_discriminator_loss.append(mean_discriminator_loss)
        overall_mean_generator_loss.append(mean_generator_loss)

    return overall_mean_generator_loss, overall_mean_discriminator_loss
