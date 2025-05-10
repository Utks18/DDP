import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
__all__ = ['CustomMLRSNetModel']

def load_clip_to_cpu(cfg):
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

class CustomMLRSNetModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

        # Prepare positive and negative prompts
        self.prompts_pos = [f"This is a {name.replace('_', ' ')}." for name in classnames]
        self.prompts_neg = [f"This is not a {name.replace('_', ' ')}." for name in classnames]

        # Tokenize prompts
        self.tokenized_prompts_pos = torch.cat([clip.tokenize(p) for p in self.prompts_pos])
        self.tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in self.prompts_neg])

        # Encode prompts (precompute for efficiency)
        with torch.no_grad():
            self.prompt_features_pos = self.text_encoder(
                clip_model.token_embedding(self.tokenized_prompts_pos).type(self.dtype),
                self.tokenized_prompts_pos
            )  # [num_classes, feat_dim]
            self.prompt_features_neg = self.text_encoder(
                clip_model.token_embedding(self.tokenized_prompts_neg).type(self.dtype),
                self.tokenized_prompts_neg
            )  # [num_classes, feat_dim]

    def forward(self, images):
        # images: [B, 3, 256, 256]
        image_features = self.image_encoder(images.type(self.dtype))  # [B, feat_dim]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Repeat prompt features for batch
        pos_feats = self.prompt_features_pos.unsqueeze(0).expand(images.size(0), -1, -1)  # [B, C, D]
        neg_feats = self.prompt_features_neg.unsqueeze(0).expand(images.size(0), -1, -1)  # [B, C, D]
        img_feats = image_features.unsqueeze(1).expand(-1, self.num_classes, -1)  # [B, C, D]

        # Contrastive logits: similarity with positive minus similarity with negative
        sim_pos = F.cosine_similarity(img_feats, pos_feats, dim=-1)  # [B, C]
        sim_neg = F.cosine_similarity(img_feats, neg_feats, dim=-1)  # [B, C]
        logits = sim_pos - sim_neg  # Higher means more likely the class is present

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
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = (torch.sigmoid(all_logits) > 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()
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
    # Dummy config and classnames
    class DummyCfg:
        class MODEL:
            class BACKBONE:
                NAME = "RN50"
            BACKBONE = BACKBONE()
        class TRAINER:
            LOGIT_SCALE_INIT = 1.0
        USE_CUDA = torch.cuda.is_available()
    cfg = DummyCfg()
    classnames = [f"class_{i}" for i in range(5)]

    # Load CLIP and model
    clip_model = load_clip_to_cpu(cfg)
    model = CustomMLRSNetModel(cfg, classnames, clip_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader
    dataset = DummyMLRSNetDataset(num_samples=100, num_classes=len(classnames))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(5):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        acc, logits, labels = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    # Visualization
    plot_results(logits, labels, classnames)