import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from pathlib import Path
import json
import numpy as np
import anndata as ad

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor

# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)

# -------------------------------------------------------
# 2. ST Encoder (single cell + spatial version)
# -------------------------------------------------------
class SCEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str = "./scGPT_CP",
        embed_dim: int = 256,
        device: str = None,
    ):
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model_dir = Path(model_dir)

        # load vocab
        vocab_file = self.model_dir / "vocab.json"
        self.vocab = GeneVocab.from_file(vocab_file)

        # load args
        with open(self.model_dir / "args.json") as f:
            args = json.load(f)

        self.pad_token=args["pad_token"] # "<pad>"
        self.pad_value=args["pad_value"] # -2

        # load model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=args["embsize"],     # 512
            nhead=args["nheads"],        # 8  
            nlayers=args["nlayers"],  # 12
            d_hid=args["d_hid"],  # 512
            vocab=self.vocab,
            dropout=args["dropout"],     # 0.2
            pad_token=self.pad_token, # "<pad>"
            pad_value=self.pad_value, # -2
            do_mvc=False,
            do_dab=False,  # inference -> 필요 없음
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            n_input_bins=args["n_bins"], # 51
            ecs_threshold=0.0,
            explicit_zero_prob=True,
        )

        ckpt_path = self.model_dir / "best_model.pt"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = {
            k: v for k, v in checkpoint.items()
            if not k.startswith("cls_decoder")
        }
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

        self.embed_dim = embed_dim
        self.proj = nn.Linear(args["embsize"], embed_dim)

    @torch.no_grad()
    def forward(self, expr: torch.Tensor):
        """
        expr: (B, G) raw/normalized counts, gene order == vocab order
        return: (B, embed_dim)
        """
        expr = expr.cpu().numpy()

        batch = tokenize_and_pad_batch(
            expr,
            np.arange(expr.shape[1]),
            max_len=30720,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value
        )

        for k, v in batch.items():
            batch[k] = v.to(self.device)
    
        out = self.model.encode_batch(
            batch["genes"], batch["values"], 
            batch["genes"] != self.vocab["<pad>"],  # mask
            batch["genes"].shape[0]
        )
    
        return self.proj(out[:, 0])  # CLS token
    
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
    Fusion options (fused_dim=256):
    - 'concat'  : concat([img, sc, st]) -> MLP -> fused (B, fused_dim)
    - 'attn'    : treat [img, sc, st] as 3 tokens -> self-attn -> pool -> fused (B, fused_dim)
    - 'sim'    : concat([img, sc, st, img*sc, img*st, sc*st, |img-sc|, |img-st|, |sc-st|, cosine(img,sc), cosine(img,st), cosine(sc,st)]) 
                -> MLP -> fused
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
                nn.Linear(embed_dim * 9 + 3, embed_dim * 2),
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
        if img_feat.ndim != 2 or st_feat.ndim != 2 or sc_feat.ndim !=2:
            raise ValueError("Input features must be 2D tensors of shape (B, D)")
        if not (img_feat.shape == sc_feat.shape == st_feat.shape):
            raise ValueError(f"Shape mismatch: {img_feat.shape} vs {sc_feat.shape} vs {st_feat.shape}")
        if not (img_feat.device == sc_feat.device == st_feat.device):
            raise ValueError(f"Device mismatch: {img_feat.device} vs {sc_feat.device} vs {st_feat.device}")
        if not (img_feat.dtype == sc_feat.dtype == st_feat.dtype):
            raise ValueError(f"Dtype mismatch: {img_feat.dtype} vs {sc_feat.dtype} vs {st_feat.dtype}")

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
                sc_n = F.normalize(sc_feat, p=2, dim=1, eps=1e-8)
                st_n = F.normalize(st_feat, p=2, dim=1, eps=1e-8)
            else:
                img_n, sc_n, st_n = img_feat, sc_feat, st_feat

             # pairwise cosine
            cos_img_sc = F.cosine_similarity(img_n, sc_n, dim=1, eps=1e-8).unsqueeze(1)
            cos_img_st = F.cosine_similarity(img_n, st_n, dim=1, eps=1e-8).unsqueeze(1)
            cos_sc_st = F.cosine_similarity(sc_n, st_n, dim=1, eps=1e-8).unsqueeze(1)

            # pairwise products
            prod_img_sc = img_n * sc_n
            prod_img_st = img_n * st_n
            prod_sc_st = sc_n * st_n

             # pairwise absolute differences
            diff_img_sc = torch.abs(img_n - sc_n)
            diff_img_st = torch.abs(img_n - st_n)
            diff_sc_st = torch.abs(sc_n - st_n)

            x = torch.cat(
                [ img_n, sc_n, st_n,
                    prod_img_sc, prod_img_st, prod_sc_st,
                    diff_img_sc, diff_img_st, diff_sc_st,
                    cos_img_sc, cos_img_st, cos_sc_st,
                ],
                dim=1,
            )
            return self.sim_fusion(x)  # (B, fused_dim)
        
        elif self.fusion_option == 'gate':
            # Concat all modality
            x = torch.cat([img_feat, sc_feat, st_feat], dim=-1)  # (B, 3D)
            weights = self.gate_fusion(x)  # (B, 3)

            # weighted sum of proj features
            proj_img = self.proj_img(img_feat)  # (B, fused_dim)
            proj_sc = self.proj_sc(sc_feat)     # (B, fused_dim)
            proj_st = self.proj_st(st_feat)     # (B, fused_dim)

            weighted = (weights[:, 0:1] * proj_img +
                    weights[:, 1:2] * proj_sc +
                    weights[:, 2:3] * proj_st)  # (B, fused_dim)
            
            return self.gate_final(weighted)

# =======================================================
# 4. MIL Attention Pooling (Spot → WSI)
# =======================================================
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim: int=None, dropout: float=0.0):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(64, embed_dim)
        
        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()    


    def forward(self, spot_embeds):
        """
        spot_embeds: (N_spots, D)
        """
        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)
        return wsi_embed, weights

# =======================================================
# 5. Layer after encoders
# =======================================================
class LinearHead(nn.Module):
    def __init__(self, dim: int, use_ln: bool=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))
       
# =======================================================
# 6. Full Multi-Modal Model (Fusion + Classifier)
# =======================================================
class MultiModalMILModel(nn.Module):
    """
      - model.img_encoder(img_batch) -> (b, D)
      - model.st_encoder(expr_batch, coord_batch) -> (b, D)   (coords ignored)
      - model.fusion(img_feat, st_feat) -> (b, D_fused)
      - model.mil_pooling(spot_embeds) -> (D_fused,), attn
      - model.classifier(wsi_embed.unsqueeze(0)) -> (1, C)
    """
    def __init__(
            self,
            num_genes: int, 
            num_classes: int=2,
            embed_dim: int=256,
            fusion_option: str='concat',
            img_backbone: str='resnet18',
            img_pretrained: bool=True,
            mil_hidden_dim: int=None,
            mil_dropout: float=0.0,
            fusion_dropout: float=0.2,
            head_use_ln: bool=True  # <- FC용 LayerNorm 사용 여부
    ):
        super().__init__()
        
        # Three encoders
        self.img_encoder = ImageEncoder(embed_dim=embed_dim, backbone=img_backbone, pretrained=img_pretrained)
        self.sc_encoder = SCEncoder(embed_dim=embed_dim)
        self.st_encoder = STEncoder(embed_dim=embed_dim)
        
        # FC head 추가
        self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        self.sc_head = nn.Identity()
        self.st_head = nn.Identity()

        # freeze
        self.freeze_encoders()
        
        print(f"Fusion option: {fusion_option}")
        
        self.fusion = FusionLayer(
            embed_dim=embed_dim,
            fusion_option=fusion_option,
            fused_dim=embed_dim,
            attn_heads=4,
            dropout=fusion_dropout,
            use_l2norm_for_sim=True,
        )
        
        # WSI-level MIL pooling over spot embeddings
        self.mil_pooling = MILAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
            dropout=mil_dropout,
        )

        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        # (Optional) hooks for XAI can be added later

    def freeze_encoders(self):
        """
        이미지 인코더 freeze / SC, ST 인코더는 유지
        주석 해제 시 SC, ST도 freeze
        """
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        # for param in self.sc_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.st_encoder.parameters():
        #     param.requires_grad = False
        
        # BN/dropout 고정
        self.img_encoder.eval()
        # self.sc_encoder.eval()
        # self.st_encoder.eval()

    def train(self, mode: bool=True):
        # Override to keep encoders in eval mode during training
        super().train(mode)
        self.img_encoder.eval()
        # self.sc_encoder.eval()
        # self.st_encoder.eval()

    def forward(self, img: torch.Tensor, expr: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        # 인코더 고정
        with torch.no_grad():
            img_feat = self.img_encoder(img)
            # sc_feat = self.st_encoder(expr)
            # st_feat = self.st_encoder(coords)
        sc_feat = self.sc_encoder(expr)         # freeze시 제거
        st_feat = self.st_encoder(coord)        # freeze시 제거

        # head만 학습
        img_feat = self.img_head(img_feat)
        sc_feat = self.sc_head(sc_feat)
        st_feat = self.st_head(st_feat)

        # Fusion
        spot_embeds = self.fusion(img_feat, sc_feat, st_feat)
        wsi_embed, attn = self.mil_pooling(spot_embeds)
        
        # Final prediction
        logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
        
        return logits, attn