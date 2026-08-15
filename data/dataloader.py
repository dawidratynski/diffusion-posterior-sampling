from collections.abc import Callable
from glob import glob

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import VisionDataset

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Callable | None=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img


@register_dataset(name='crystal')
class CrystalDataset(VisionDataset):
    '''EM crystal samples produced by the data_processing pipeline.

    Images on disk are 150x150 RGB PNGs. UVCGAN2 is trained on them at 160x160
    (`shape: (3, 160, 160)` with a `resize` to 160), so we resize here too: the
    diffusion prior and the CycleGAN operator must live at the same resolution
    and in the same [-1, 1] pixel convention, otherwise the DPS data-consistency
    term compares incompatible tensors.

    Args:
        root: directory of PNGs, searched recursively.
        image_size: target resolution. Must match the UVCGAN2 generator.
        augment: random flips + 90-degree rotations. The structure is a lattice,
            so these are label-preserving. Use for training, not for evaluation.
        transforms: applied after the resize. Defaults to ToTensor + Normalize
            to [-1, 1].
    '''

    def __init__(self,
                 root: str,
                 image_size: int = 160,
                 augment: bool = False,
                 transforms: Callable | None = None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, f"No PNGs found under {root}."
        self.image_size = image_size

        # Bilinear to match torchvision's default, which is what UVCGAN2's
        # 'resize' transform uses.
        steps = [T.Resize((image_size, image_size),
                          interpolation=T.InterpolationMode.BILINEAR)]
        if augment:
            steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]

        steps.append(self.transforms if self.transforms is not None
                     else T.Compose([
                         T.ToTensor(),
                         T.Normalize((0.5,) * 3, (0.5,) * 3),
                     ]))
        self.pipeline = T.Compose(steps)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        img = Image.open(self.fpaths[index]).convert('RGB')
        return self.pipeline(img)