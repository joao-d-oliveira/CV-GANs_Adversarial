import logging
import sys

import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision import transforms
from helpers.Config import Config


def init_logger(name='simple_example', filename=None):
    """ Initialise a logger

    Parameters
    ----------
    name : str
        The name of the logger
    filename : str
        The filename to save the logger to

    Returns
    -------
    logger : logging.Logger
        The logger
    """
    # create logger
    logger = logging.getLogger(name)

    if filename is None:
        # create console handler and set level to debug
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # add formatter to stdout_handler
        stdout_handler.setFormatter(formatter)

        # add stdout_handler to logger
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
    else:
        logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info(f'Started logger named: {name}')

    return logger


def print_time(duration):
    """Print elapsed time in a user-friendly format HH:MM:SS

    Parameters
    ----------
    duration : float
        Duration to which convert

    Returns
    -------
    string : str
        String of duration converted into format DD day(s) HH:MM:SS
    """
    d = int(duration // (60 * 60 * 24))
    duration -= d * (60 * 60 * 24)
    h = int(duration // (60 * 60))
    duration -= h * (60 * 60)
    m = int(duration // 60)
    duration -= m * 60
    s = int(duration)
    duration -= s
    ms = f"{duration:.2f}".split('.')[-1]

    return f"{str(d).zfill(2)} day(s) {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2):}.{ms}"


def print_real_fake_images(config: Config, real, fake, i):
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

    if config.store_local_images:
        plt.savefig(f'images/output_{i}.png')

    if config.store_wandb_images:
        plt.savefig(f'/tmp/output_{i}.png')
        wandb.log({f'Output Epoch {i}': wandb.Image(f'/tmp/output_{i}.png')})

    plt.close()


def get_dataloader(config: Config):
    # Load MNIST dataset as tensors using DataLoader class
    dataloader = DataLoader(
        MNIST(config.data_folder, download=True, transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=True)

    return dataloader
