import time
import torch
import wandb
from helpers.Discriminator import Discriminator
from helpers.Generator import Generator
from helpers.Config import Config
from helpers.gans import train
from helpers.utils import get_dataloader, print_time
from helpers.w_and_b import record_losses_plot


if __name__ == '__main__':
    start = time.time()

    config = Config()
    dataloader = get_dataloader(config)

    # Generator & Optimizer for Generator
    gen = Generator(config.noise_dimension).to(config.device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=config.lr)

    # Discriminator & Optimizer for Discriminator
    disc = Discriminator().to(config.device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=config.lr)

    mean_generator_loss, mean_discriminator_loss = train(config, dataloader, gen, gen_opt, disc, disc_opt)

    if config.wandb_project:
        record_losses_plot(mean_generator_loss, "mean_generator_loss", "Mean Generator Loss Plot")
        record_losses_plot(mean_discriminator_loss, "mean_discriminator_loss", "Mean Discriminator Loss Plot")
        wandb.finish()

    print(f'Total time taken: {print_time(time.time() - start)}')
