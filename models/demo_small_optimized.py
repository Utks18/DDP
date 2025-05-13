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

# Initialize tokenizer
_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    """Load CLIP model to CPU memory."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class OptimizedMLRSNetModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)
        
        # Create prompts
        self.prompts_pos = [f"This is a {name.replace('_', ' ')}." for name in classnames]
        self.prompts_neg = [f"This is not a {name.replace('_', ' ')}." for name in classnames]

        # Tokenize prompts
        self.tokenized_prompts_pos = torch.cat([clip.tokenize(p) for p in self.prompts_pos])
        self.tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in self.prompts_neg])

        # Precompute prompt features
        with torch.no_grad():
            text_embeddings_pos = clip_model.token_embedding(self.tokenized_prompts_pos).type(self.dtype)
            text_embeddings_neg = clip_model.token_embedding(self.tokenized_prompts_neg).type(self.dtype)
            
            prompt_features_pos = self.text_encoder(text_embeddings_pos, self.tokenized_prompts_pos)
            prompt_features_neg = self.text_encoder(text_embeddings_neg, self.tokenized_prompts_neg)
            
            prompt_features_pos = prompt_features_pos / prompt_features_pos.norm(dim=-1, keepdim=True)
            prompt_features_neg = prompt_features_neg / prompt_features_neg.norm(dim=-1, keepdim=True)
            
            self.register_buffer('prompt_features_pos', prompt_features_pos)
            self.register_buffer('prompt_features_neg', prompt_features_neg)

    def forward(self, images):
        # Get image features
        image_features = self.image_encoder(images.type(self.dtype))
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        
        # Average pool spatial dimensions if needed
        if len(image_features.shape) > 2:
            image_features = image_features.mean(dim=[-2, -1])
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Reshape image features to match prompt features
        B = image_features.shape[0]  # Batch size
        image_features = image_features.view(B, -1)  # Reshape to [B, D]
        
        # Compute similarities using matrix multiplication
        sim_pos = image_features @ self.prompt_features_pos.T  # [B, C]
        sim_neg = image_features @ self.prompt_features_neg.T  # [B, C]
        
        return sim_pos - sim_neg

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = (torch.sigmoid(all_logits) > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()
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
    # Configuration setup
    class DummyCfg:
        class MODEL:
            class BACKBONE:
                NAME = "RN50"
            BACKBONE = BACKBONE()
        class TRAINER:
            LOGIT_SCALE_INIT = 1.0
        class INPUT:
            SIZE = [224, 224]
        USE_CUDA = torch.cuda.is_available()
    cfg = DummyCfg()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    images_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Images"
    labels_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Labels"
    categories_file = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Categories_names.xlsx"

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match CLIP's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset with smaller sample fraction
    dataset = MLRSNetSmallDataset(
        images_dir, 
        labels_dir, 
        categories_file, 
        split="train", 
        transform=transform,
        sample_fraction=0.001  # Use 0.1% of data for faster testing
    )
    
    # Create dataloader with optimized settings
    dataloader = DataLoader(
        dataset, 
        batch_size=64,  # Larger batch size for better GPU utilization
        shuffle=True,
        num_workers=2,  # Reduced workers to prevent memory issues
        pin_memory=True
    )

    # Load CLIP and create model
    clip_model = load_clip_to_cpu(cfg)
    model = OptimizedMLRSNetModel(cfg, dataset.categories, clip_model)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # Reduced epochs
        print(f"\nEpoch {epoch+1}/2")
        loss = train_one_epoch(model, dataloader, optimizer, device)
        acc, logits, labels = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1} - Loss={loss:.4f}, Accuracy={acc:.4f}")

    # Visualization
    plot_results(logits, labels, dataset.categories) 