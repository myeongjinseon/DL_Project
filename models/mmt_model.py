"""
Multimodal Machine Translation Model

구조:
  1. Text-only Transformer (baseline)
  2. + Image fusion: 이미지 피처를 linear projection 후
     encoder 입력 시퀀스 앞에 추가 토큰으로 prepend

fusion 방식은 동일하게 유지하고, 이미지 피처 종류(ResNet/CLIP)만 교체하여 비교합니다.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ImageProjection(nn.Module):
    """이미지 피처 벡터를 d_model 차원으로 projection"""
    def __init__(self, img_feat_dim, d_model, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(img_feat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, img_features):
        # img_features: (batch, img_feat_dim)
        # output: (batch, 1, d_model) — encoder 시퀀스 앞에 prepend할 1개 토큰
        projected = self.projection(img_features)  # (batch, d_model)
        return projected.unsqueeze(1)  # (batch, 1, d_model)


class MMTModel(nn.Module):
    """
    Multimodal Machine Translation Transformer

    Args:
        src_vocab_size: source vocabulary 크기
        tgt_vocab_size: target vocabulary 크기
        d_model: 모델 차원
        n_heads: attention head 수
        n_encoder_layers: encoder layer 수
        n_decoder_layers: decoder layer 수
        d_ff: feedforward 차원
        dropout: dropout 비율
        pad_idx: padding token index
        img_feat_dim: 이미지 피처 차원 (None이면 text-only)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256,
                 n_heads=8, n_encoder_layers=4, n_decoder_layers=4,
                 d_ff=512, dropout=0.3, pad_idx=0, img_feat_dim=None):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_image = img_feat_dim is not None

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Image projection (multimodal 모드에서만 사용)
        if self.use_image:
            self.img_projection = ImageProjection(img_feat_dim, d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying (embedding ↔ output projection)
        # tgt_embedding과 output_proj의 weight를 공유하면 파라미터 효율성 증가
        # (vocab size가 같을 때만 적용하거나, 별도로 적용)

        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src, has_img_token=False):
        """
        Source padding mask 생성
        src: (batch, src_len)
        이미지 토큰이 prepend된 경우, 해당 위치는 mask하지 않음
        """
        # (batch, src_len) → True where padded
        src_pad_mask = (src == self.pad_idx)

        if has_img_token:
            # 이미지 토큰 위치 (항상 첫 번째)는 mask하지 않음
            batch_size = src_pad_mask.size(0)
            img_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
            src_pad_mask = torch.cat([img_mask, src_pad_mask], dim=1)

        return src_pad_mask

    def _make_tgt_mask(self, tgt):
        """Target causal mask + padding mask"""
        tgt_len = tgt.size(1)
        # Causal mask: (tgt_len, tgt_len) upper triangular
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()
        return causal_mask

    def _make_tgt_pad_mask(self, tgt):
        """Target padding mask"""
        return (tgt == self.pad_idx)

    def encode(self, src, img_features=None):
        """
        Encoder forward
        src: (batch, src_len)
        img_features: (batch, img_feat_dim) or None
        Returns: encoder output (batch, seq_len, d_model), src_pad_mask
        """
        # Source embedding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # (batch, src_len, d_model)
        src_emb = self.pos_encoding(src_emb)

        has_img = False
        if self.use_image and img_features is not None:
            # 이미지 피처를 projection하여 시퀀스 앞에 prepend
            img_token = self.img_projection(img_features)  # (batch, 1, d_model)
            src_emb = torch.cat([img_token, src_emb], dim=1)  # (batch, 1+src_len, d_model)
            has_img = True

        # Source mask
        src_pad_mask = self._make_src_mask(src, has_img_token=has_img)

        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    def decode(self, tgt, memory, memory_key_padding_mask=None):
        """
        Decoder forward
        tgt: (batch, tgt_len)
        memory: encoder output
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        tgt_mask = self._make_tgt_mask(tgt)
        tgt_pad_mask = self._make_tgt_pad_mask(tgt)

        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output

    def forward(self, src, tgt, img_features=None):
        """
        Full forward pass
        src: (batch, src_len) — source token IDs
        tgt: (batch, tgt_len) — target token IDs (teacher forcing)
        img_features: (batch, img_feat_dim) or None

        Returns: logits (batch, tgt_len, tgt_vocab_size)
        """
        # Encode
        memory, src_pad_mask = self.encode(src, img_features)

        # Decode (teacher forcing: 입력은 tgt[:-1], 예측 대상은 tgt[1:])
        tgt_input = tgt[:, :-1]
        decoder_output = self.decode(tgt_input, memory, memory_key_padding_mask=src_pad_mask)

        # Output projection
        logits = self.output_proj(decoder_output)  # (batch, tgt_len-1, tgt_vocab_size)
        return logits

    @torch.no_grad()
    def translate(self, src, img_features=None, max_len=64, bos_idx=1, eos_idx=2):
        """
        Greedy decoding으로 번역
        src: (1, src_len) — 단일 문장
        Returns: token ID 리스트
        """
        self.eval()
        device = src.device

        memory, src_pad_mask = self.encode(src, img_features)

        # 시작 토큰
        ys = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        for _ in range(max_len):
            decoder_output = self.decode(ys, memory, memory_key_padding_mask=src_pad_mask)
            logits = self.output_proj(decoder_output[:, -1, :])  # 마지막 토큰의 logits
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

        return ys.squeeze(0).tolist()


def build_model(src_vocab_size, tgt_vocab_size, feature_type=None, config=None):
    """모델 생성 헬퍼"""
    if config is None:
        from configs.config import Config as config

    img_feat_dim = None
    if feature_type:
        img_feat_dim = config.IMG_FEATURE_DIM[feature_type]

    model = MMTModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_encoder_layers=config.N_ENCODER_LAYERS,
        n_decoder_layers=config.N_DECODER_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        pad_idx=0,
        img_feat_dim=img_feat_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built: {feature_type or 'text-only'}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,}")
    if img_feat_dim:
        print(f"  Image feat dim: {img_feat_dim}")

    return model
