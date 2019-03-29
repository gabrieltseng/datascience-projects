import torch
from torchvision import datasets, transforms


class NoiseMaker:
    """Makes the inputs to the generator
    """
    def __init__(self, device, num_channels=40, noise_size=7, batch_size=64):

        self.device = device

        self.num_channels = num_channels
        self.noise_size = noise_size
        self.batch_size = batch_size

    def __call__(self):

        return torch.empty(self.batch_size, self.num_channels,
                           self.noise_size, self.noise_size,
                           device=self.device).normal_(0, 1)


def get_mnist_vals(data_location):
    # Download the MNIST dataset (if its not already downloaded), and return its
    # mean and std

    # we use the MNIST dataset from pytorch since it automatically downloads
    # the data for us if its not present
    mnist_ds = datasets.MNIST(data_location, train=True, download=True,
                              transform=transforms.ToTensor())
    ims = []
    for i, _ in mnist_ds:
        ims.append(i)
    ims = torch.stack(ims)
    return ims.mean().item(), ims.std().item()
