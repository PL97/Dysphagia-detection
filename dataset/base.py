import torch
import pandas as pd
import os
import random
import torchvision
import itertools
from torchvision.transforms import Compose

def get_samples(root, mode):
    df = pd.read_csv(os.path.join(root, mode+".csv"))
    target = list(df.label)
    path = list(root.replace("annotations/", "") + df.split + "/" + df.youtube_id + "_" +\
                df.time_start.astype(str).str.zfill(6) + "_" + \
                df.time_end.astype(str).str.zfill(6) + ".mp4")
    return list(zip(path, target))
    
    
    

class BaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16, mode='train'):
        super(BaseDataset).__init__()

        self.samples = get_samples(root, mode)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer
            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                if self.frame_transform != None:
                    video_frames.append(self.frame_transform(frame['data']))
                else:
                    video_frames.append(frame['data'])
                current_pts = frame['pts']
            
            # Stack it into a tensor
            video = torch.stack(video_frames, 0).permute(1, 0, 2, 3)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output
            
            
if __name__ == "__main__":
    # pass
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
    transforms = Compose([
        ToPILImage(),
        CenterCrop(64),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ])
    test = BaseDataset(root="/data/datasets/kinetics-dataset/k400/annotations/", mode="val", frame_transform=transforms, clip_len=3)
    for x in test:
        print(x['video'].shape)
        exit()