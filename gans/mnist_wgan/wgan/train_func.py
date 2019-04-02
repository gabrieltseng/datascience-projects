import torch
import numpy as np
from tqdm import tqdm


def train_wgan_epoch(discriminator, generator, dataloader, noisemaker, ncritic, device):
    # This training loop is taken from the WGAN paper (2017)

    num_batches = len(dataloader)

    i = 0
    discriminator_real = []
    discriminator_fake = []
    generator_loss = []

    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)

    discriminator.train(), generator.train()
    with tqdm(total=num_batches) as pbar:
        while i <= num_batches:
            # first, we train the discriminator
            for _ in range(ncritic):
                d_optimizer.zero_grad(), g_optimizer.zero_grad()

                real, _ = next(iter(dataloader))
                if device.type != 'cpu': real = real.cuda()

                fake = noisemaker()
                with torch.no_grad():
                    fake_images = generator(fake)
                real_preds = discriminator(real)
                fake_preds = discriminator(fake_images)
                loss = real_preds - fake_preds
                loss.backward()
                discriminator_real.append(real_preds.item())
                discriminator_fake.append(fake_preds.item())
                d_optimizer.step()
                discriminator.clamp_weights(min=-0.01, max=0.01)
                i += 1
                pbar.update(1)

            # Next, we train the generator
            d_optimizer.zero_grad(), g_optimizer.zero_grad()

            fake = noisemaker()
            fake_images = generator(fake)
            loss = discriminator(fake_images)
            generator_loss.append(loss.item())
            loss.backward()
            g_optimizer.step()

    print(f"Finished epoch! "
          f"Discriminator predictions for real images: {np.mean(discriminator_real)}, "
          f"Discriminator predictions for fake images: {np.mean(discriminator_fake)}, "
          f"Generator loss: {np.mean(generator_loss)}")


def train_ganhacks_epoch(discriminator, generator, dataloader, noisemaker, device, ncritic=None):
    # This training loop adopts all the best practices described in
    # https://github.com/soumith/ganhacks
    # The most significant difference is to seperate the real and fake images in each minibatch
    # from the discriminator

    # note that ncritic does nothing here, but makes it easier to swap between both training methods
    discriminator_real = []
    discriminator_fake = []
    generator_loss = []

    # "optim.Adam rules!"
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    discriminator.train(), generator.train()
    for real, _ in tqdm(dataloader):
        # first, we train the discriminator
        for minibatch_type in ['real', 'fake']:
            d_optimizer.zero_grad(), g_optimizer.zero_grad()

            if minibatch_type == 'real':
                real, _ = next(iter(dataloader))
                if device.type != 'cpu': real = real.cuda()
                real_preds = discriminator(real)
                discriminator_real.append(real_preds.item())
                loss = real_preds
            else:
                fake = noisemaker()
                with torch.no_grad():
                    fake_images = generator(fake)
                fake_preds = discriminator(fake_images)
                discriminator_fake.append(fake_preds.item())
                loss = - fake_preds
            loss.backward()
            d_optimizer.step()

        # next, we train the generator
        d_optimizer.zero_grad(), g_optimizer.zero_grad()

        fake = noisemaker()
        fake_images = generator(fake)
        loss = discriminator(fake_images)
        generator_loss.append(loss.item())
        loss.backward()
        g_optimizer.step()

    print(f"Finished epoch! "
          f"Discriminator predictions for real images: {np.mean(discriminator_real)}, "
          f"Discriminator predictions for fake images: {np.mean(discriminator_fake)}, "
          f"Generator loss: {np.mean(generator_loss)}")
