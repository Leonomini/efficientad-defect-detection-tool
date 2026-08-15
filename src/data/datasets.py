from pathlib import Path
import cv2
import numpy as np
from torchvision.datasets import ImageFolder


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

class ImageFolderWithGT(ImageFolderWithPath):
    def __init__(self, root: Path):
        super().__init__(root)

        self.dataset_root = Path(root).parent
        # self.split = Path(root).name  # "test" oder "val"

        self.gt_root = self.dataset_root / "ground_truth" # / self.split

        if not self.gt_root.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: {self.gt_root}"
            )

    def __getitem__(self, index):
        image, target, path = super().__getitem__(index)
        path = Path(path)

        rel_path = path.relative_to(self.root)
        gt_path = self.gt_root / rel_path

        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

            if gt is None:
                raise RuntimeError(f"GT mask exists but could not be read: {gt_path}")
        else:
            # keine GT vorhanden => normales Sample => Nullmaske
            w, h = image.size
            gt = np.zeros((h, w), dtype=np.uint8)

        return image, gt, path




def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
