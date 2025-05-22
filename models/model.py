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
__all__ = ['CustomMLRSNetModel', 'mlrsmodel']

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    # model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)
    model = clip.build_model(state_dict or model.state_dict())


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

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(
            self,
            query,
            data,
            mask=None,
    ):
        b, *_, device = *data.shape, data.device
        # x = repeat(soft_prompt, 'n d -> b n d', b=b)
        x = query
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x)+x

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
        self.clip = clip_model
        self.cross_attention = nn.ModuleList([CrossAttention(512,512) for _ in range(12)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

        # Create positive and negative prompts for each class
        self.prompts_pos = [f"This is a {name.replace('_', ' ')}." for name in classnames]
        self.prompts_neg = [f"This is not a {name.replace('_', ' ')}." for name in classnames]

        # Tokenize prompts
        self.tokenized_prompts_pos = torch.cat([clip.tokenize(p) for p in self.prompts_pos])
        self.tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in self.prompts_neg])

        # Precompute prompt features for efficiency
        with torch.no_grad():
            self.prompt_features_pos = clip_model.encode_text(self.tokenized_prompts_pos).type(self.dtype)
            self.prompt_features_neg = clip_model.encode_text(self.tokenized_prompts_neg).type(self.dtype)
        
        print("Prompt pos features shape:", self.prompt_features_pos.shape)
        print("Prompt neg features shape:", self.prompt_features_neg.shape)

    def forward_clip(self, image):
        with torch.no_grad():

            # image_features = self.clip.encode_image(image)
            batch_size = 32  # Define your desired batch size
            image_features = []
            for i in range(0, image.shape[0], batch_size):
                batch = image[i:i + batch_size]  # Create a batch
                image_features.append(self.clip.encode_image(batch))

            image_features = torch.cat(image_features)


        image_features = image_features/image_features.norm(dim=1, keepdim=True)
        
        return image_features
        
    def forward(self, images, cls_id=None):
        B = images.shape[0]
        image_features = self.forward_clip(images)
        
        cross_attended_outputs_pos = self.prompt_features_pos.unsqueeze(0).repeat(B,1,1).to(images.device)
        for i in range(len(self.cross_attention)):
            cross_attended_outputs_pos = self.cross_attention[i](cross_attended_outputs_pos, image_features.unsqueeze(1))
        
        cross_attended_outputs_neg = self.prompt_features_neg.unsqueeze(0).repeat(B,1,1).to(images.device)
        for i in range(len(self.cross_attention)):
            cross_attended_outputs_neg = self.cross_attention[i](cross_attended_outputs_neg, image_features.unsqueeze(1))
       
        pos_logits = torch.sigmoid(self.classifier(cross_attended_outputs_pos).squeeze(-1))
        neg_logits = torch.sigmoid(self.classifier(cross_attended_outputs_neg).squeeze(-1))

        return pos_logits, neg_logits


    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if ("image_encoder" in name or 'text_encoder' in name or 'clip' in name) and "cross" not in name and 'class' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'cross_attention' in name:
                params.append(param)
                print(name)
        return params
    
    def classifier_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'classifier' in name:
                params.append(param)
                print(name)
        return params

def mlrsmodel(cfg, classnames, **kwargs):
    
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building CustomMLRSNetModel")
    model = CustomMLRSNetModel(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    # if not cfg.TRAINER.FINETUNE_ATTN:
    #     print('Freeze the attn weights')
    #     attn_params = model.attn_params()
    #     for param in attn_params:
    #         param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    # device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
    #     model = nn.DataParallel(model)
    return model