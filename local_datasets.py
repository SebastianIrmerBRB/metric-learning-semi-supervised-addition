import os
import zipfile

from pytorch_metric_learning.datasets.sop import StanfordOnlineProducts as _StanfordOnlineProducts
from pytorch_metric_learning.utils.common_functions import _urlretrieve
from torchvision.datasets import CIFAR10 as _CIFAR10


class CIFAR10(_CIFAR10):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        super().__init__(
            root=root,
            train=split == "train",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.split = split
        self.labels = list(self.targets)

### Standford neu erstellen, da auf Windows bspw. kein im Link erlaubt ist
class StanfordOnlineProducts(_StanfordOnlineProducts):
    FILE_ID = "1TclrpQOF_ullUP99wk_gjGN8pKvtErG8"
    ARCHIVE_NAME = "Stanford_Online_Products.zip"

    def download_and_remove(self):
        os.makedirs(self.root, exist_ok=True)

        extracted_root = os.path.join(self.root, "Stanford_Online_Products")
        train_file = os.path.join(extracted_root, "Ebay_train.txt")
        test_file = os.path.join(extracted_root, "Ebay_test.txt")

        if os.path.exists(train_file) and os.path.exists(test_file):
            return

        download_file_path = os.path.join(self.root, "Stanford_Online_Products.zip")

        _urlretrieve(
            url=StanfordOnlineProducts.DOWNLOAD_URL,
            filename=download_file_path,
        )

        try:
            with zipfile.ZipFile(download_file_path, "r") as zip_ref:
                zip_ref.extractall(self.root)
        finally:
            if os.path.exists(download_file_path):
                os.remove(download_file_path)
