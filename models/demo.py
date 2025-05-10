# Standard library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
import os

# Add project root to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import CLIP and its tokenizer
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dataloaders.MLRSNet import MLRSNetDataset
from torchvision import transforms

# Initialize tokenizer and define module exports
_tokenizer = _Tokenizer()
__all__ = ['CustomMLRSNetModel']

def load_clip_to_cpu(cfg):
    """
    Load CLIP model to CPU memory.
    Args:
        cfg: Configuration object containing model settings
    Returns:
        CLIP model with convolutional projection
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # Try loading as TorchScript model
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # Fallback to loading state dict
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)
    return model

class TextEncoder(nn.Module):
    """
    Text encoder component that processes text prompts using CLIP's transformer.
    """
    def __init__(self, clip_model):
        super().__init__()
        # Extract text encoding components from CLIP
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Forward pass for text encoding.
        Args:
            prompts: Text prompt embeddings
            tokenized_prompts: Tokenized text prompts
        Returns:
            Text features
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class CustomMLRSNetModel(nn.Module):
    """
    Main model class that combines CLIP's image and text encoders for multi-label classification.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # Initialize encoders
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

        # Create positive and negative prompts for each class
        self.prompts_pos = [f"This is a {name.replace('_', ' ')}." for name in classnames]
        self.prompts_neg = [f"This is not a {name.replace('_', ' ')}." for name in classnames]

        # Tokenize prompts
        self.tokenized_prompts_pos = torch.cat([clip.tokenize(p) for p in self.prompts_pos])
        self.tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in self.prompts_neg])

        # Precompute prompt features for efficiency
        with torch.no_grad():
            self.prompt_features_pos = self.text_encoder(
                clip_model.token_embedding(self.tokenized_prompts_pos).type(self.dtype),
                self.tokenized_prompts_pos
            )
            self.prompt_features_neg = self.text_encoder(
                clip_model.token_embedding(self.tokenized_prompts_neg).type(self.dtype),
                self.tokenized_prompts_neg
            )

    def forward(self, images):
        """
        Forward pass for the model.
        Args:
            images: Batch of input images [B, 3, H, W]
        Returns:
            Logits for each class
        """
        # Get image features
        image_features = self.image_encoder(images.type(self.dtype))
        if isinstance(image_features, tuple):
            image_features = image_features[0]  # Handle tuple output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Prepare features for similarity computation
        pos_feats = self.prompt_features_pos.unsqueeze(0).expand(images.size(0), -1, -1)
        neg_feats = self.prompt_features_neg.unsqueeze(0).expand(images.size(0), -1, -1)
        img_feats = image_features.unsqueeze(1).expand(-1, self.num_classes, -1)

        # Compute similarities and contrastive logits
        sim_pos = F.cosine_similarity(img_feats, pos_feats, dim=-1)
        sim_neg = F.cosine_similarity(img_feats, neg_feats, dim=-1)
        logits = sim_pos - sim_neg
        return logits

# --- Example Dataset and DataLoader ---

class DummyMLRSNetDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=5, img_size=256):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy data: random image and random multilabels
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, 2, (self.num_classes,)).float()
        return img, label

# --- Training Loop ---

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        device: Device to train on (CPU/GPU)
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    total_batches = len(dataloader)
    
    print(f"\nStarting epoch training...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{total_batches}] - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch completed - Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, device):
    """
    Evaluate the model on the validation/test set.
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on (CPU/GPU)
    Returns:
        Accuracy, logits, and labels
    """
    model.eval()
    all_logits, all_labels = [], []
    total_batches = len(dataloader)
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluation Batch [{batch_idx + 1}/{total_batches}]")
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = (torch.sigmoid(all_logits) > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()
    print(f"Evaluation completed - Accuracy: {accuracy:.4f}")
    return accuracy, all_logits, all_labels

# --- Visualization ---

def plot_results(logits, labels, classnames, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, logits.size(0))):
        plt.subplot(2, 5, i+1)
        plt.bar(range(len(classnames)), torch.sigmoid(logits[i]).numpy())
        plt.title(f"True: {labels[i].int().numpy()}")
        plt.xticks(range(len(classnames)), classnames, rotation=90)
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# --- Usage Example ---

if __name__ == "__main__":
    # Configuration setup
    class DummyCfg:
        class MODEL:
            class BACKBONE:
                NAME = "RN50"  # Use ResNet-50 as backbone
            BACKBONE = BACKBONE()
        class TRAINER:
            LOGIT_SCALE_INIT = 1.0
        class INPUT:
            SIZE = [224, 224]  # Default CLIP input size
        USE_CUDA = torch.cuda.is_available()
    cfg = DummyCfg()

    # Paths to dataset components
    images_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Images"
    labels_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Labels"
    categories_file = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Categories_names.xlsx"

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = MLRSNetDataset(images_dir, labels_dir, categories_file, split="train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Get classnames from dataset
    classnames = dataset.categories

    # Load CLIP and create model
    clip_model = load_clip_to_cpu(cfg)
    model = CustomMLRSNetModel(cfg, classnames, clip_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        loss = train_one_epoch(model, dataloader, optimizer, device)
        acc, logits, labels = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    print("\nTraining completed!")

    # Visualization
    plot_results(logits, labels, classnames)