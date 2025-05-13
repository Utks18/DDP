# Standard library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import time

# Add project root to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import CLIP and its tokenizer
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dataloaders.MLRSNetSmall import MLRSNetSmallDataset
from torchvision import transforms

# Initialize tokenizer and define module exports
_tokenizer = _Tokenizer()
__all__ = ['CustomMLRSNetModel']

def print_status(message):
    """Print a status message with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

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
        print(f"Number of classes: {self.num_classes}")

        # Create positive and negative prompts for each class
        self.prompts_pos = [f"This is a {name.replace('_', ' ')}." for name in classnames]
        self.prompts_neg = [f"This is not a {name.replace('_', ' ')}." for name in classnames]

        # Tokenize prompts
        self.tokenized_prompts_pos = torch.cat([clip.tokenize(p) for p in self.prompts_pos])
        self.tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in self.prompts_neg])
        print(f"Tokenized prompts shape: {self.tokenized_prompts_pos.shape}")

        # Precompute prompt features for efficiency
        with torch.no_grad():
            # Get text embeddings
            text_embeddings_pos = clip_model.token_embedding(self.tokenized_prompts_pos).type(self.dtype)
            text_embeddings_neg = clip_model.token_embedding(self.tokenized_prompts_neg).type(self.dtype)
            print(f"Text embeddings shape: {text_embeddings_pos.shape}")
            
            # Encode text features
            prompt_features_pos = self.text_encoder(text_embeddings_pos, self.tokenized_prompts_pos)
            prompt_features_neg = self.text_encoder(text_embeddings_neg, self.tokenized_prompts_neg)
            print(f"Prompt features shape: {prompt_features_pos.shape}")
            
            # Normalize features
            prompt_features_pos = prompt_features_pos / prompt_features_pos.norm(dim=-1, keepdim=True)
            prompt_features_neg = prompt_features_neg / prompt_features_neg.norm(dim=-1, keepdim=True)
            
            # Register as buffers to ensure device compatibility
            self.register_buffer('prompt_features_pos', prompt_features_pos)
            self.register_buffer('prompt_features_neg', prompt_features_neg)

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
        print(f"Raw image features shape: {image_features.shape}")
        
        # Average pool spatial dimensions if needed
        if len(image_features.shape) > 2:
            image_features = image_features.mean(dim=[-2, -1])
        print(f"Pooled image features shape: {image_features.shape}")
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Reshape image features to match prompt features
        image_features = image_features.unsqueeze(1)  # [B, 1, D]
        prompt_features_pos = self.prompt_features_pos.unsqueeze(0)  # [1, C, D]
        prompt_features_neg = self.prompt_features_neg.unsqueeze(0)  # [1, C, D]
        
        # Compute similarities using cosine similarity
        sim_pos = F.cosine_similarity(image_features, prompt_features_pos, dim=-1)  # [B, C]
        sim_neg = F.cosine_similarity(image_features, prompt_features_neg, dim=-1)  # [B, C]
        
        # Compute final logits
        logits = sim_pos - sim_neg  # [B, C]
        print(f"Logits shape: {logits.shape}")
        return logits

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
    
    print_status("Starting epoch training...")
    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc="Training")
    
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        
        logits = model(images)
        print(f"Model output shape: {logits.shape}")
        
        # Check dimensions match
        assert logits.shape[1] == labels.shape[1], f"Logits shape {logits.shape} doesn't match labels shape {labels.shape}"
        
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader.dataset)
    print_status(f"Epoch completed - Average Loss: {avg_loss:.4f}")
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
    
    print_status("Starting evaluation...")
    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in progress_bar:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
            # Update progress bar
            progress_bar.set_postfix({'batch': f'{batch_idx + 1}/{total_batches}'})
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = (torch.sigmoid(all_logits) > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()
    print_status(f"Evaluation completed - Accuracy: {accuracy:.4f}")
    return accuracy, all_logits, all_labels

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

if __name__ == "__main__":
    print_status("Starting program execution...")
    
    # Configuration setup
    print_status("Setting up configuration...")
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
    print_status("Setting up dataset paths...")
    images_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Images"
    labels_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Labels"
    categories_file = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Categories_names.xlsx"

    # Define image transformations
    print_status("Setting up image transformations...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    print_status("Loading dataset (this might take a while)...")
    dataset = MLRSNetSmallDataset(
        images_dir, 
        labels_dir, 
        categories_file, 
        split="train", 
        transform=transform,
        sample_fraction=0.005  # Using 0.5% of the data for faster testing
    )
    print_status(f"Dataset loaded with {len(dataset)} images")
    
    # Increase batch size for faster processing
    dataloader = DataLoader(
        dataset, 
        batch_size=32,  # Increased from 16 to 32
        shuffle=True,
        num_workers=4,  # Add parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    print_status(f"Created dataloader with {len(dataloader)} batches")

    # Get classnames from dataset
    classnames = dataset.categories
    print_status(f"Number of classes: {len(classnames)}")

    # Load CLIP and create model
    print_status("Loading CLIP model...")
    clip_model = load_clip_to_cpu(cfg)
    print_status("Creating custom model...")
    model = CustomMLRSNetModel(cfg, classnames, clip_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_status(f"Using device: {device}")
    model.to(device)

    # Setup optimizer with higher learning rate for faster convergence
    print_status("Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Doubled learning rate

    # Training loop with fewer epochs
    print_status("Starting training loop...")
    for epoch in range(3):  # Reduced from 5 to 3 epochs
        print_status(f"\nEpoch {epoch+1}/3")
        loss = train_one_epoch(model, dataloader, optimizer, device)
        acc, logits, labels = evaluate(model, dataloader, device)
        print_status(f"Epoch {epoch+1} completed - Loss={loss:.4f}, Accuracy={acc:.4f}")

    print_status("Training completed!")

    # Visualization
    print_status("Generating visualization...")
    plot_results(logits, labels, classnames)
    print_status("Program execution completed!") 