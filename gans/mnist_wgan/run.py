import torch
from torchvision import datasets, transforms
import numpy as np

from wgan import Generator, Discriminator, train_epoch
from wgan.utils import NoiseMaker, get_mnist_vals


def main(batch_size=64, num_epochs=100, save_preds=True):

    data_dir = 'mnist_data'
    mean, std = get_mnist_vals(data_location=data_dir)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([mean], [std])
                       ])),
        batch_size=batch_size, shuffle=True)
    noisemaker = NoiseMaker(batch_size=batch_size)

    discriminator = Discriminator()
    generator = Generator()

    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-5)
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-5)

    for i in range(num_epochs):
        print(f"Epoch number {i + 1}")
        train_epoch(discriminator, generator, d_optimizer, g_optimizer, dataloader,
                    noisemaker, ncritic=100 if ((i == 0) or (i == 10)) else 5)

    # save some predictions
    if save_preds:
        with torch.no_grad():
            generator.eval()
            noise = noisemaker()
            output = generator(noise).numpy()
            # denormalize
            output = (output * std) + mean
            np.save("generator_output.npy", output)


if __name__ == '__main__':
    main()