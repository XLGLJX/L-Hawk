import os
import torchvision as tv
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader


class Background(Dataset):
    def __init__(self, path, size=640, subset_size=None) -> None:
        super().__init__()
        self.img_names = sorted(glob(os.path.join(path, "*.png")))
        if len(self.img_names) == 0:
            self.img_names = sorted(glob(os.path.join(path, "*.jpg")))
        if len(self.img_names) == 0:
            raise FileNotFoundError(f"No .png or .jpg images found in evaluation directory: {path}")
        if subset_size is not None and subset_size > 0:
            self.img_names = self.img_names[:subset_size]
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.CenterCrop((size, size)),
            tv.transforms.ToTensor()
        ])
        self.process = lambda x: self.transform(Image.open(x))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self.process(img_name)
        return img

def load_kitti(cfg, batch_size=1, size=640, shuffle=True, mask=True, subset_size=None, seed=0):
    path = cfg.DATA.TEST.KITTI_DIR
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        torch.manual_seed(seed + worker_id)

    loader = DataLoader(
        Background(path, size=size, subset_size=subset_size),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return loader

def load_bdd100k(cfg, batch_size=1, size=640, shuffle=True, subset_size=None, seed=0):
    path = cfg.DATA.TEST.BDD100K_DIR
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        torch.manual_seed(seed + worker_id)

    loader = DataLoader(
        Background(path, size=size, subset_size=subset_size),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return loader
