from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_size=28, start_channels=10, additional_blocks=2):
        super().__init__()

        layers = [ConvBlock(1, start_channels, 4, 2, 1)]
        input_size = input_size // 2

        for i in range(additional_blocks):
            layers.append(ConvBlock(start_channels, start_channels, 3, 1, 1))

        while input_size > 4:
            layers.append(ConvBlock(start_channels, start_channels // 2, 4, 2, 1))

            input_size = input_size // 2
            start_channels = start_channels // 2

        layers.append(nn.Conv2d(start_channels, 1, 3))
        self.layers = nn.Sequential(*layers)

    def clamp_weights(self, min=-0.01, max=0.01):
        for layer in self.parameters():
            layer.data.clamp_(min=min, max=max)

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1).mean()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 negative_slope=0.2):
        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        return self.convblock(x)
