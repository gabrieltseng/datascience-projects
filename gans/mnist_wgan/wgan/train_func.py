import torch
import numpy as np
from tqdm import tqdm


def train_epoch(discriminator, generator, d_optimizer, g_optimizer, dataloader,
                noisemaker, ncritic, device):

    num_batches = len(dataloader)

    i = 0
    discriminator_real = []
    discriminator_fake = []
    generator_loss = []

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
