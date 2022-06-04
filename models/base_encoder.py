import torch
import torchvision.models as models


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        self.feature_extractor = torch.nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool,

            self.resnet18.layer1,
            self.resnet18.layer2,
            self.resnet18.layer3,
            self.resnet18.layer4,

            self.resnet18.avgpool
        )
        self.last_layer_dim = 512

    def forward(self, x):
        return torch.flatten(self.feature_extractor(x), 1)
