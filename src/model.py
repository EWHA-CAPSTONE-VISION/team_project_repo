import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# -------------------------------------------------------
# 1. Image Encoder (CNN or ViT alternative)
# -------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Use lightweight ResNet18 (can replace with ViT if needed)
        self.backbone = models.resnet18(pretrained=True)
        # Replace final classification layer with embed_dim output
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        return self.backbone(x)  # -> (B, embed_dim)

# -------------------------------------------------------
# 2. ST Encoder (single cell + spatial version)
# -------------------------------------------------------
class SCEncoder(nn.Module):
    def __init__(self, num_genes, embed_dim=256):
        super().__init__()
        # Convert raw gene expression vector → embedding
        self.gene_embed = nn.Sequential(
            nn.Linear(num_genes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Transformer Encoder (core structure of scBERT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final projection layer
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, num_genes)
        x = self.gene_embed(x)  # -> (B, embed_dim)

        # Add sequence dimension for transformer: (B, 1, embed_dim)
        x = x.unsqueeze(1)

        # Apply transformer
        x = self.transformer(x)  # -> (B, 1, embed_dim)

        # Remove sequence dimension
        x = x.squeeze(1)

        return self.fc(x)
    
class STEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Convert (x, y) -> embedding
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, embed_dim//2),  # 2D -> D/2
            nn.LayerNorm(embed_dim//2),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_enc = nn.Sequential(
            nn.Linear(2, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.Tanh()   # range: [-1, 1]           
        )

        # CLS token + spatial token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
            dropout = 0.1,
            activation = 'gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Final projection layer
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: (B, 2) / spatial codinates (x, y)
        B = x.shape[0]

        spatial_emb = self.spatial_embed(x)   # (B, D/2)
        pos_emb = self.pos_enc(x)             # (B, D/2)
        token_emb = torch.cat([spatial_emb, pos_emb], dim=-1)

        # CLS token + spatial token
        cls = self.cls_token.expand(B, 1, -1)                 # (B, 1, D)
        seq = torch.cat([cls, token_emb.unsqueeze(1)], dim=1) # (B, 2, D)

        # Apply transformer
        seq = self.transformer(seq) # (B, 2, D)

        # Apply CLS token
        seq = seq[:,0,:]            # (B, D)

        return self.fc(seq)

# -------------------------------------------------------
# 3. Fusion Layer
# -------------------------------------------------------
class FusionLayer(nn.Module):
    """
    Fusion options (fused_dim=128):
    - 'concat'  : concat([img, sc, st]) -> MLP -> fused (B, fused_dim)
    - 'attn'    : treat [img, sc, st] as 3 tokens -> self-attn -> pool -> fused (B, fused_dim)
    - 'sim'     : concat([img, st, img*st, |img-st|, cosine(img, st)]) -> MLP -> fused (B, fused_dim) / only for 2-encoder fusion
    - 'gate'    : gate ([img, sc, st]) -> MLP -> fused (B, fused_dim)
    """
    def __init__(
            self,
            embed_dim: int=256,
            fusion_option: str='concat',
            fused_dim: int=128,
            attn_heads: int=4,
            dropout: float=0.2,
            use_l2norm_for_sim: bool=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.fused_dim = fused_dim
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim
        
        # pre-norm for modality feature alignment
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_sc = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':
            self.concat_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, fused_dim*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 2, fused_dim)
            )

        elif fusion_option == 'attn':
            self.attn_fusion = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(embed_dim)    # post-attn norm (residual)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),    # (D, 4D)
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),    # (4D, D)
                nn.Dropout(dropout)
            )
            self.norm2 = nn.LayerNorm(embed_dim)    # 2nd norm after FFN residual
            self.out_proj = nn.Linear(embed_dim, fused_dim) # output=128

        elif fusion_option == 'sim':
            self.sim_fusion = nn.Sequential(
                nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        elif fusion_option == 'gate':
            self.gate_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3),  # gate weights for 3 modalities
                nn.Softmax(dim=-1)
            )
            self.proj_img = nn.Linear(embed_dim, fused_dim)
            self.proj_sc = nn.Linear(embed_dim, fused_dim)
            self.proj_st = nn.Linear(embed_dim, fused_dim)
            self.gate_final = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        else:
            raise ValueError(f"Unknown fusion option={fusion_option}")

    def forward(self, img_feat: torch.Tensor, sc_feat: torch.Tensor, st_feat: torch.Tensor) -> torch.Tensor:
        """
        img_feat: (B, D)
        sc_feat : (B, D)
        st_feat : (B, D)
        return  : (B, 128)
        """

        # basic checks
        if img_feat.ndim != 2 or st_feat.ndim != 2:
            raise ValueError("Input features must be 2D tensors of shape (B, D)")
        if img_feat.shape != st_feat.shape:
            raise ValueError(f"Shape mismatch: {img_feat.shape} vs {st_feat.shape}")
        if img_feat.device != st_feat.device:
            raise ValueError(f"Device mismatch: {img_feat.device} vs {st_feat.device}")
        if img_feat.dtype != st_feat.dtype:
            raise ValueError(f"Dtype mismatch: {img_feat.dtype} vs {st_feat.dtype}")

        # pre-norm to align distributions
        img_feat = self.pre_norm_img(img_feat)
        sc_feat = self.pre_norm_sc(sc_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':
            x = torch.cat([img_feat, sc_feat,st_feat], dim=1)   # (B, 3D)
            return self.concat_fusion(x)                # (B, fused_dim)
        
        elif self.fusion_option == 'attn':
            # 3 tokens: [img_feat, sc_feat, st_feat]
            tokens = torch.stack([img_feat, sc_feat, st_feat], dim=1)  # (B, 3, D)
            
            # Self-attn block + residual + norm
            attn_out, _ = self.attn_fusion(tokens, tokens, tokens)  # (B, 3, D)
            tokens = self.norm1(tokens + attn_out)

            # FFN block + residual + norm
            ffn_out = self.ffn(tokens)  # (B, 3, D)
            tokens = self.norm2(tokens + ffn_out)

            pooled = tokens.mean(dim=1)     # (B, D)
            return self.out_proj(pooled)    # (B, fused_dim)
        
        elif self.fusion_option == 'sim':
            # Optionally L2-normalize features before cosine similarity
            if self.use_l2norm_for_sim:
                img_n = F.normalize(img_feat, p=2, dim=1, eps=1e-8)
                st_n = F.normalize(st_feat, p=2, dim=1, eps=1e-8)
            else:
                img_n = img_feat
                st_n = st_feat
            sim = F.cosine_similarity(img_n, st_n, dim=1, eps=1e-8).unsqueeze(1)  # (B, 1)
            prod = img_n * st_n                 # (B, D)
            abs_diff = torch.abs(img_n - st_n)  # (B, D)

            x = torch.cat([img_n, st_n, prod, abs_diff, sim], dim=1)  # (B, 4D+1)
            return self.sim_fusion(x)  # (B, fused_dim)
        
        elif self.fusion_option == 'gate':
            # Concat all modality
            input = torch.cat([img_feat, sc_feat, st_feat], dim=-1)  # (B, 3D)
            weights = self.gate_fusion(input)  # (B, 3)

            # weighted sum of proj features
            proj_img = self.proj_img(img_feat)  # (B, fused_dim)
            proj_sc = self.proj_sc(sc_feat)     # (B, fused_dim)
            proj_st = self.proj_st(st_feat)     # (B, fused_dim)

            weighted = (weights[:, 0:1] * proj_img +
                    weights[:, 1:2] * proj_sc +
                    weights[:, 2:3] * proj_st)  # (B, fused_dim)
            
            return self.gate_final(weighted)


# -------------------------------------------------------
# 4. Full Multi-Modal Model (Fusion + Classifier)
# -------------------------------------------------------
class MultiModalHestModel(nn.Module):
    def __init__(
            self,
            num_genes: int, 
            num_classes: int=2,
            fusion_option: str='concat',
    ):
        super().__init__()
        embed_dim = 256
        
        # Three encoders
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.sc_encoder = SCEncoder(num_genes=num_genes, embed_dim=embed_dim)
        self.st_encoder = STEncoder(embed_dim=embed_dim)
        
        print(f"Fusion option: {fusion_option}")
        
        self.fusion = FusionLayer(
            embed_dim=embed_dim,
            fusion_option=fusion_option,
            fused_dim=128,
            attn_heads=4,
            dropout=0.2
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)

        # (Optional) hooks for XAI can be added later

    def forward(self, img: torch.Tensor, expr: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        # Encode each modality
        img_feat = self.img_encoder(img)      # (B, 256)
        sc_feat = self.sc_encoder(expr)       # (B, 256)
        st_feat = self.st_encoder(coord)      # (B, 256)
        
        # Fusion module
        fused = self.fusion(img_feat, sc_feat, st_feat)

        # Final prediction
        logits = self.classifier(fused)                   # (B, num_classes)
        
        return logits