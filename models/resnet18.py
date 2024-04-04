import pytorch_lightning as pl

class ResNet18(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)