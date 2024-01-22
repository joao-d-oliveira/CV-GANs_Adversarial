import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader
from helpers.Discriminator import Discriminator
from helpers.Generator import  Generator
from helpers.gans import get_noise
from helpers.Config import Config
from helpers.gans import get_disc_loss, get_gen_loss


def get_dataloader(config: Config):
    # Load MNIST dataset as tensors using DataLoader class
    dataloader = DataLoader(
        MNIST(config.data_folder, download=True, transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=True)

    return dataloader


def print_images(real, fake, i):
    num_images = 30
    size = (1, 28, 28)

    image_unflat_real = real.detach().cpu().view(-1, *size)
    image_grid_real = make_grid(image_unflat_real[:num_images], nrow=6)
    image_grid_real = image_grid_real.permute(1, 2, 0).squeeze()

    image_unflat_fake = fake.detach().cpu().view(-1, *size)
    image_grid_fake = make_grid(image_unflat_fake[:num_images], nrow=6)
    image_grid_fake = image_grid_fake.permute(1, 2, 0).squeeze()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure and a set of subplots

    # Display the first image
    axs[0].imshow(image_grid_real)
    axs[0].set_title('Real Images')

    # Display the second image
    axs[1].imshow(image_grid_fake)
    axs[1].set_title('Fake Images')

    # Remove the axes
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'images/output_{i}.png')  # replace 'output.png' with your preferred filename and path
    plt.close()


def train(config:Config, dataloader:DataLoader, gen:Generator, gen_opt:torch.optim, disc:Discriminator, disc_opt:torch.optim):
    cur_step, mean_generator_loss, mean_discriminator_loss = 0, 0, 0

    n = len(dataloader)
    print("DataLoader length: ", n)

    for epoch in tqdm(range(config.n_epochs)):
        print(f"\nEpoch {epoch} / {config.n_epochs}")

        cur_batch_size = 0
        for i, data in enumerate(tqdm(dataloader)):
            # Get number of batch size (number of image)
            # And get tensor for each image in batch
            real, _ = data
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(config.device)

            ### Traing discriminator ###
            # Zero out the gradient .zero_grad()
            # Calculate discriminator loss get_disc_loss()
            # Update gradient .gradient()
            # Update optimizer .step()
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, config.criterion, real, cur_batch_size, config.noise_dimension, config.device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            ### Traing generator ###
            # Zero out the gradient .zero_grad()
            # Calculate discriminator loss get_gen_loss()
            # Update gradient .gradient()
            # Update optimizer .step()
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, config.criterion, cur_batch_size, config.noise_dimension, config.device)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / config.display_step
            mean_generator_loss += gen_loss.item() / config.display_step

            cur_step += 1

            # at the end of the epoch print the images
            if i == n - 1:
                fake_noise = get_noise(cur_batch_size, config.noise_dimension, device=config.device)
                fake = gen(fake_noise)

                print_images(real, fake, epoch)
                mean_generator_loss, mean_discriminator_loss = 0, 0


if __name__ == '__main__':
    _config = Config()
    _dataloader = get_dataloader(_config)

    # Generator & Optimizer for Generator
    _gen = Generator(_config.noise_dimension).to(_config.device)
    _gen_opt = torch.optim.Adam(_gen.parameters(), lr=_config.lr)

    # Discriminator & Optimizer for Discriminator
    _disc = Discriminator().to(_config.device)
    _disc_opt = torch.optim.Adam(_disc.parameters(), lr=_config.lr)

    # Binary Cross Entropy Loss
    train(_config, _dataloader, _gen, _gen_opt, _disc, _disc_opt)
