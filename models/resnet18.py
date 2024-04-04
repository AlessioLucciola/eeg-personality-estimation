import pytorch_lightning as pl

class ResNet18(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #TO DO: Define model architecture