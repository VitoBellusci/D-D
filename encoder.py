from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model = resnet50, w = ResNet50_Weights):
        super().__init__()

        self.embed_size = embed_size
        resenet = model(weights=w.DEFAULT)
        modules = list(resenet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.linear = nn.Linear(resenet.fc.in_features, embed_size)

    def forward(self, images):
        features_ext = self.backbone(images)
        features_shaped = features_ext.view(features_ext.shape[0], -1)

        return self.linear(features_shaped)[:-1]