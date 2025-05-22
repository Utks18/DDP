import os
from os.path import join
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MLRSNetDataset(Dataset):
    def __init__(self, root,  split="train", img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        """
        Args:
            images_dir (str): Path to the folder containing images.
            labels_dir (str): Path to the folder containing label CSV files.
            categories_file (str): Path to the Categories_names.xlsx file.
            split (str): Dataset split to use ("train", "val", or "test").
            transform (callable, optional): Transformations to apply to the images.
        """
        self.images_dir = join(root, "Images")
        self.labels_dir = join(root, "labels")
        self.categories_file = join(root, 'Categories_names.xlsx')
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load category names and multi-label mappings
        self.categories = pd.read_excel(self.categories_file, sheet_name="Categories")["Categories"].tolist()
        self.num_classes = len(self.categories)
        self.classnames = None
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        """
        Load image paths and corresponding multi-label annotations.
        """
        image_paths = []
        labels = []

        # Iterate through each category's label CSV file
        for category in self.categories:
            label_file = os.path.join(self.labels_dir, f"{category}.csv")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")

            # Read the CSV file
            df = pd.read_csv(label_file)
            if self.classnames is None:
                self.classnames = list(df.columns)[1:]
            for _, row in df.iterrows():
                image_name = row["image"]
                label = row[1:].values.astype(float)  # Multi-labels (binary)

                # Append image path and label
                image_path = os.path.join(self.images_dir, category, category, image_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(torch.tensor(label, dtype=torch.float32))
                else:
                    print(f"Warning: Image not found: {image_path}")

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding multi-label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def name(self):
        return 'MLRSNet'


# Example usage
if __name__ == "__main__":
    # Paths to dataset components
    images_dir = r"/home/sarthak/DDP/MLRSNet/Images"
    labels_dir = r"/home/sarthak/DDP/MLRSNet/labels"
    categories_file = r"/home/sarthak/DDP/MLRSNet/Categories_names.xlsx"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = MLRSNetDataset(images_dir, labels_dir, categories_file, split="train", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through the dataloader
    for images, labels in dataloader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")