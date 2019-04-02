import torch
from torchvision import datasets, transforms
import numpy as np

from wgan import Generator, Discriminator, train_wgan_epoch, train_ganhacks_epoch
from wgan.utils import NoiseMaker, get_mnist_vals


def main(batch_size=64, num_epochs=3, save_preds=True, train_method='ganhacks'):

    str2method = {
        'wgan': train_wgan_epoch,
        'ganhacks': train_ganhacks_epoch
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    data_dir = 'mnist_data'
    mean, std = get_mnist_vals(data_location=data_dir)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([mean], [std])
                       ])),
        batch_size=batch_size, shuffle=True)
    noisemaker = NoiseMaker(batch_size=batch_size, device=device)

    discriminator, generator = Discriminator(), Generator()

    if device.type != 'cpu':
        discriminator, generator = discriminator.cuda(), generator.cuda()

    for i in range(num_epochs):
        print(f"Epoch number {i + 1}")
        # ncritic is only meaningful if train_method == 'wgan'
        str2method[train_method](discriminator, generator, dataloader, noisemaker,
                                 ncritic=100 if ((i == 0) or (i == 10)) else 5,
                                 device=device)

    # save some predictions
    if save_preds:
        with torch.no_grad():
            generator.eval()
            noise = noisemaker()
            output = generator(noise).cpu().numpy()
            # denormalize
            output = (output * std) + mean
            np.save("generator_output.npy", output)


if __name__ == '__main__':
    main()
