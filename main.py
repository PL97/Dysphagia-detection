import sys
import pytorch_lightning
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    ToPILImage,
    Compose,
    CenterCrop,
    ToTensor,
    Normalize,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

sys.path.append('.')

from dataset.kinetics import KineticsDataModule
from dataset.base import BaseDataset
from model.models import VideoClassificationLightningModule
from model.models import make_kinetics_resnet
 
def train():
    root="/data/datasets/kinetics-dataset/k400/annotations/"
    transforms = Compose([
        ToPILImage(),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ])
    classification_module = VideoClassificationLightningModule()
    # data_module = KineticsDataModule()
    data_module = DataLoader(BaseDataset(root=root, mode='train', frame_transform=transforms, clip_len=16), \
                            batch_size=4,
                            num_workers=1)
    
    ## for debug only
    # model = make_kinetics_resnet()
    
    # for x in data_module:
    #     print(x['video'].shape)
    #     print(model(x['video']).shape)
    #     exit()
    
    trainer = pytorch_lightning.Trainer(max_epochs=10,
                                        accelerator='gpu',
                                        devices=1)
    trainer.fit(classification_module, data_module)
    
if __name__ == "__main__":
    train()