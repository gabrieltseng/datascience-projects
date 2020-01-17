import torch
from torch import nn
from torch.nn import functional as F


class MaskedConv(nn.Conv2d):

    """A masked conv layer described in https://arxiv.org/pdf/1601.06759.pdf

    The mask type is described in Figure 2, Right.

    Also, padding is set so that the size of the output is preserved.
    """

    def __init__(
        self, mask_type: str, in_channels: int, out_channels: int, kernel_size: int = 7
    ) -> None:

        assert mask_type in {
            "A",
            "B",
        }, f"Expected mask type to be one of 'A' or 'B', got {mask_type}"
        assert (
            in_channels >= 3
        ), f"Channels should be >= 3, got in_channels = {in_channels}"
        assert (
            out_channels >= 3
        ), f"Channels should be >= 3, got in_channels = {out_channels}"

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=None,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )

        self.mask = self.set_mask(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            mask_type=mask_type,
        )

    @staticmethod
    def set_mask(
        kernel_size: int, in_channels: int, out_channels: int, mask_type: str
    ) -> torch.Tensor:
        mask = torch.ones((out_channels, in_channels, kernel_size, kernel_size)).float()

        # assumes the kernel size is odd
        midpoint = kernel_size // 2

        # The weight matrix is in_channels, out_channels, H, W
        mask[:, :, midpoint + 1 :, :] = 0
        mask[:, :, midpoint, midpoint + 1 :] = 0

        # also, the channels. We will group the channels by 3,
        # RGBRGBRGBRGB.... so that all Rs get no dependencies, all Gs get Rs and
        # all Bs get Rs and Gs
        channel_group = [0, 1, 2]
        for idx in channel_group:
            up_to = idx
            if mask_type == "B":
                up_to += 1

            # construct the indices
            out_indices = [idx]
            while out_indices[-1] + 3 < out_channels:
                out_indices.append(out_indices[-1] + 3)

            in_channel_group = channel_group[up_to:]
            in_indices = [i for i in in_channel_group if i < in_channels]
            if len(in_channel_group) > 0:
                while in_channel_group[-1] < in_channels:
                    in_channel_group = [i + 3 for i in in_channel_group]
                    in_indices.extend([i for i in in_channel_group if i < in_channels])

                for out_idx in out_indices:
                    mask[out_idx, in_indices, midpoint, midpoint] = 0

        return mask

    def conv2d_same_padding(self, input, weight):
        # stride and dilation are expected to be tuples.

        # first, we'll figure out how much padding is necessary for the rows
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(
            0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows - input_rows
        )
        rows_odd = padding_rows % 2 != 0

        # same for columns
        input_cols = input.size(3)
        filter_cols = weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(
            0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols - input_cols
        )
        cols_odd = padding_cols % 2 != 0

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_same_padding(x, self.weight * self.mask)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        assert channels % 2 == 0, f"channels should be divisible by 2, got {channels}"

        self.conv_layers = nn.Sequential(
            *[
                nn.ReLU(),
                MaskedConv(
                    mask_type="B",
                    in_channels=channels,
                    out_channels=channels // 2,
                    kernel_size=1,
                ),
                nn.ReLU(),
                MaskedConv(
                    mask_type="B",
                    in_channels=channels // 2,
                    out_channels=channels // 2,
                    kernel_size=3,
                ),
                nn.ReLU(),
                MaskedConv(
                    mask_type="B",
                    in_channels=channels // 2,
                    out_channels=channels,
                    kernel_size=1,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.conv_layers(x)


class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = MaskedConv("A", in_channels=3, out_channels=128, kernel_size=7)

        self.resnet_blocks = nn.Sequential(*[ResBlock(channels=128)] * 12)

        self.out_layers = nn.Sequential(
            *[
                nn.ReLU(),
                MaskedConv("B", in_channels=128, out_channels=64, kernel_size=1),
                nn.ReLU(),
                MaskedConv("B", in_channels=64, out_channels=12, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.in_conv(x)
        x = self.resnet_blocks(x)
        return self.out_layers(x)
