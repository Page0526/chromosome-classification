import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL.TiffImagePlugin import TiffImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_images(main_folder):
    image_data = []

    for chrom in range(0, 25):
        chrom_folder = os.path.join(main_folder, str(chrom))

        if not os.path.isdir(chrom_folder):
            continue

        # Get all TIFF files in the subfolder
        tiff_files = [f for f in os.listdir(chrom_folder) if f.lower().endswith(('.tiff', '.tif'))]

        for file in tiff_files:
            file_path = os.path.join(chrom_folder, file)
            image = Image.open(file_path)
            image_data.append((image, chrom))

    return image_data

class ChromosomeDataset(Dataset):
    def __init__(self, folder, ims_list=None, transform=None):
        self.image_data = load_images(folder) if ims_list is None else ims_list
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        im, label = self.image_data[idx]

        if self.transform:
            im = self.transform(im)
        return im, label - 1

def visualize(ims_list, num=25):
    display = min(len(ims_list), num)
    cols = int(np.ceil(np.sqrt(display)))
    rows = int((np.ceil(display / cols)))
    plt.figure(figsize=(cols * 2, rows * 2))
    indices = None
    if len(ims_list) > 0:
        indices = random.sample(range(len(ims_list)), display)
        
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        im, label = ims_list[idx]

        # if tensor, convert to numpy
        if hasattr(im, 'detach'):
            im = im.detach().cpu().numpy()
        if isinstance(im, TiffImageFile):
            im = np.array(im)

        # (C, H, W) to (H, W, C)
        if im.shape == 3 and im.shape[0] in [1, 3]:
            im=np.transpose(im, (1, 2, 0))

        # normalize
        if im.max() > 1.0:
            im = im/255.0

        plt.imshow(im, cmap='gray')
        plt.title(str(label))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    path = 'D:\\VSC\\chromosome-classification\\data\\train_samples'
    ims = load_images(path)
    print(f"Loaded {len(ims)} TIFF images")
    visualize(ims)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # EfficientNet normalization
    ])
    dataset = ChromosomeDataset(path, ims, transform=transform)