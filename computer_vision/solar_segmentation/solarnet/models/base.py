from torch import nn
from torchvision.models import resnet34


class ResnetBase(nn.Module):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model
    """
    def __init__(self, add_forward_hooks=False, imagenet_base=True):
        super().__init__()

        resnet = resnet34(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

        if add_forward_hooks:
            self.hooks = self.add_hooks()

    def add_hooks(self):
        hooks = []
        target_modules = [str(x) for x in [2, 4, 5, 6, 7]]
        for name, child in self.pretrained:
            if name in target_modules:
                hooks.append(child.register_forward_hook(self.save_output))
        return hooks

    def load_base(self, state_dict):
        # This allows a model trained on the classifier to be loaded
        # into the model used for segmentation, even though their state_dicts
        # differ
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def save_output(module, input, output):
        module.output = output

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError
