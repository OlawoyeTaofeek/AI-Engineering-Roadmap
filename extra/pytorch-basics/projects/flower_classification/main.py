import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import scipy.io
import requests
import tarfile
from PIL import Image
from tqdm.auto import tqdm
import time

def download_dataset():
    """
    Downloads and extracts a dataset from remote URLs if not already present locally.

    This function first checks for the existence of the dataset files in a specific
    directory. If the files are not found, it proceeds to download them from
    pre-defined URLs, showing progress bars, and then extracts the contents.
    """

    data_dir = "flower_data"
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

    os.makedirs(data_dir, exist_ok=True)

    image_path = os.path.join(data_dir, "102flowers.tgz")
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    if os.path.exists(image_path) and os.path.exists(labels_path):
        print(f"Dataset already exists locally. Loading from '{data_dir}'. Skipping download.")
        return

    # Create the data directory if it doesn't exist.
    print("Dataset not found locally. Starting download...")

    response = requests.get(image_url, stream=True)
    # Get the total size of the file
    total_size = int(response.headers.get("content-length", 0))
    with open(image_path, "wb") as file, tqdm(
        desc="Downloading Images",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            progress_bar.update(len(chunk))

    extract_dataset(image_path, data_dir)

    print("Downloading labels")
    response = requests.get(labels_url)
    with open(labels_path, "wb") as file:
        # Write the entire content of the response to the file.
        file.write(response.content)


    # create labels_description.txt
    labels_description = [
        'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily',
        'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea',
        'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
        'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
        'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation',
        'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
        'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
        'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia',
        'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
        'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
        'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily',
        'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
        'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress',
        'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
        'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily'
    ]

    with open(os.path.join(data_dir, "labels_description.txt"), "w") as f:
        for idx, label in enumerate(labels_description, start=1):
            f.write(f"{label}\n")



def extract_dataset(image_path, extract_to="flower_data"):
    """Extract the downloaded .tgz file into the target directory"""
    print(f"Extracting {image_path}...")
    with tarfile.open(image_path, "r:gz") as tar:
        tar.extractall(extract_to)
    print(f"Extraction complete. Files are in '{extract_to}/'")

## getting the tree structure of the folder
def tree(dir_path, prefix=""):

    files = os.listdir(dir_path)
    print("flower_data/")
    for i, file in enumerate(files):

        path = os.path.join(dir_path, file)

        connector = "└── " if i == len(files) - 1 else "├── "

        print(prefix + connector + file)

        
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),
    ## Standard Preprocessing
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RobustFlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_dir = os.path.join(root_dir, "jpg")
        label_mat = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))
        self.labels = label_mat['labels'][0] - 1  # convert 1-indexed -> 0-indexed

        self.error_log = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = None
        try:
            img_name = f"image_{idx+1:05d}.jpg"
            img_path = os.path.join(self.img_dir, img_name)

            image = Image.open(img_path)
            image.verify()               # .verify() closes the file handle
            image = Image.open(img_path)  # so it must be reopened to actually use it

            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError(f"Image too small: {image.size}")
            if image.mode != "RGB":
                image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

            label = int(self.labels[idx])
            return image, label

        except Exception as e:
            self.error_log.append({
                "index": idx,
                "error": str(e),
                "path": img_path if img_path is not None else "Unknown"
            })
            print(f"Warning, skipping corrupted image {idx}: {e}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

    def error_summary(self):
        if not self.error_log:
            print("No errors encountered - dataset is too clean")
        else:
            print(f"\nEncountered {len(self.error_log)} problematic images")
            for error in self.error_log[:5]:
                print(f"    Index {error['index']}: {error['error']}")
            if len(self.error_log) > 5:
                print(f"..... and {len(self.error_log) - 5} more")


class TransformedSubset(Dataset):
    """Wraps a Subset (or any Dataset) and applies its own transform,
    independent of whatever transform the underlying dataset has."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  # raw PIL image, since base dataset has transform=None
        if self.transform:
            image = self.transform(image)
        return image, label

def split_dataset(root_dir, train_transform, eval_transform,
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split the flower dataset, applying train_transform only to train
    and eval_transform only to val/test."""

    base = RobustFlowerDataset(root_dir, transform=None)  # loaded ONCE

    n = len(base)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split, test_split = random_split(
        base, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = TransformedSubset(train_split, train_transform)
    val_dataset = TransformedSubset(val_split, eval_transform)
    test_dataset = TransformedSubset(test_split, eval_transform)

    print(f"Dataset split -> train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}, test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                        batch_size=32, num_workers=2, pin_memory=False):
    """
    Wrap train/val/test datasets in DataLoaders.

    - train: shuffled, drop_last=True (avoids a tiny, unstable final batch
      during training, e.g. for BatchNorm statistics)
    - val/test: not shuffled, drop_last=False (evaluate on every sample)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

class MonitoredDataset(RobustFlowerDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)
        self.access_counts = {}
        self.load_times = []

    def __getitem__(self, idx):
        start_time = time.time()
        self.access_counts[idx] = self.access_counts.get(idx, 0) + 1

        result = super().__getitem__(idx)

        load_time = time.time() - start_time
        self.load_times.append(load_time)

        if load_time > 1.0:
            print(f"Slow load: Image {idx} took {load_time:.2f}s")

        return result

    def timing_summary(self):
        if not self.load_times:
            print("No images loaded yet.")
            return

        avg_time = sum(self.load_times) / len(self.load_times)
        max_time = max(self.load_times)
        min_time = min(self.load_times)

        print(f"Loaded {len(self.load_times)} images")
        print(f"  avg load time: {avg_time*1000:.2f} ms")
        print(f"  min load time: {min_time*1000:.2f} ms")
        print(f"  max load time: {max_time*1000:.2f} ms")

    def most_accessed(self, top_n=5):
        if not self.access_counts:
            print("No accesses recorded yet.")
            return
        sorted_counts = sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {top_n} most-accessed indices:")
        for idx, count in sorted_counts[:top_n]:
            print(f"  index {idx}: accessed {count} times")

def visualize_augmentation(dataset, idx=0, num_version=8):
    ...