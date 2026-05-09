"""
MMT Training Script

사용법:
    python train.py --mode text_only
    python train.py --mode resnet
    python train.py --mode clip
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config
from utils.vocab import Vocabulary
from utils.dataset import get_dataloader, prepare_vocabs
from models.mmt_model import build_model


class WarmupScheduler:
    """Transformer 학습에 사용되는 warmup learning rate scheduler"""
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_image):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        img = batch["img"].to(device) if use_image and "img" in batch else None

        # Forward: logits for tgt[:-1], labels are tgt[1:]
        logits = model(src, tgt, img_features=img)  # (batch, tgt_len-1, vocab_size)
        labels = tgt[:, 1:]  # (batch, tgt_len-1)

        # Loss 계산 (reshape for cross entropy)
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()
        lr = scheduler.step()

        # 통계 (padding 제외)
        non_pad = (labels != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else 0


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_image):
    """Validation loss 계산"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        img = batch["img"].to(device) if use_image and "img" in batch else None

        logits = model(src, tgt, img_features=img)
        labels = tgt[:, 1:]

        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss = criterion(logits, labels)

        non_pad = (labels != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else 0


def train(mode="text_only"):
    """
    전체 학습 파이프라인

    Args:
        mode: "text_only", "resnet", "clip"
    """
    device = Config.DEVICE
    feature_type = None if mode == "text_only" else mode

    print("=" * 60)
    print(f"Training MMT Model — Mode: {mode}")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Vocab 준비
    src_vocab_path = os.path.join(Config.DATA_DIR, "vocab_en.json")
    tgt_vocab_path = os.path.join(Config.DATA_DIR, "vocab_de.json")

    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        print(f"Loaded vocab — EN: {len(src_vocab)}, DE: {len(tgt_vocab)}")
    else:
        src_vocab, tgt_vocab = prepare_vocabs()

    # 2. DataLoader
    print("\nLoading data...")
    train_loader = get_dataloader("train", src_vocab, tgt_vocab, feature_type,
                                  batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader("val", src_vocab, tgt_vocab, feature_type,
                                batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # 3. 모델
    print("\nBuilding model...")
    model = build_model(len(src_vocab), len(tgt_vocab), feature_type).to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(
        ignore_index=src_vocab.pad_idx,
        label_smoothing=Config.LABEL_SMOOTHING,
    )
    optimizer = Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, Config.D_MODEL, Config.WARMUP_STEPS)

    # 5. Training loop
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    use_image = feature_type is not None

    print(f"\nStarting training for max {Config.MAX_EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_image)
        val_loss = evaluate(model, val_loader, criterion, device, use_image)

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Checkpoint (best model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{mode}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mode": mode,
            }, ckpt_path)
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={Config.PATIENCE})")
                break

    print("-" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="text_only",
                        choices=["text_only", "resnet", "clip"],
                        help="Training mode")
    args = parser.parse_args()

    train(args.mode)
