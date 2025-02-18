import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimized image transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Reduced size for speed
        transforms.ToTensor(),
    ]
)


# Custom dataset for loading images without using ImageFolder
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(("jpg", "jpeg", "png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert(
            "RGB"
        )  # Open image and ensure it's in RGB mode
        if self.transform:
            image = self.transform(image)
        return image


# Fast image loading using the custom dataset
def load_images(image_dir, batch_size=32):
    dataset = CustomImageDataset(image_dir, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Move images to GPU and convert to half precision for speed
    images_tensor = torch.cat([batch.to(device).half() for batch in dataloader])
    return images_tensor


if __name__ == "__main__":
    # Define image directory and load images
    image_dir = "data/dataset/images/train"
    processed_images = load_images(image_dir)

    # Verify the shape of the output tensor
    print(f"Processed images shape: {processed_images.shape}")
