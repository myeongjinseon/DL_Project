"""
Multi30K Dataset: 텍스트 + (선택적) 이미지 피처 로딩
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config
from utils.vocab import Vocabulary


class Multi30KDataset(Dataset):
    def __init__(self, split, src_vocab, tgt_vocab, feature_type=None, max_len=64):
        """
        Args:
            split: "train", "val", "test"
            src_vocab: source Vocabulary
            tgt_vocab: target Vocabulary
            feature_type: None (text-only), "resnet", or "clip"
            max_len: 최대 시퀀스 길이
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.feature_type = feature_type
        self.max_len = max_len

        # 텍스트 로드
        split_info = Config.SPLITS[split]
        src_path = os.path.join(Config.DATA_DIR, split_info["en"])
        tgt_path = os.path.join(Config.DATA_DIR, split_info["de"])

        with open(src_path, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f if line.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f if line.strip()]

        assert len(self.src_sentences) == len(self.tgt_sentences)

        # 이미지 피처 로드 (미리 추출된 .npy 파일)
        self.img_features = None
        if feature_type:
            feat_path = os.path.join(Config.FEATURE_DIR, f"{split}_{feature_type}.npy")
            if os.path.exists(feat_path):
                self.img_features = np.load(feat_path)
                assert len(self.img_features) == len(self.src_sentences), \
                    f"Feature count mismatch: {len(self.img_features)} vs {len(self.src_sentences)}"
                print(f"  Loaded {feature_type} features: {self.img_features.shape}")
            else:
                print(f"  WARNING: {feat_path} not found. Using zero features.")

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(self.src_sentences[idx], max_len=self.max_len)
        tgt_ids = self.tgt_vocab.encode(self.tgt_sentences[idx], max_len=self.max_len)

        item = {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }

        if self.feature_type:
            if self.img_features is not None:
                item["img"] = torch.tensor(self.img_features[idx], dtype=torch.float32)
            else:
                dim = Config.IMG_FEATURE_DIM[self.feature_type]
                item["img"] = torch.zeros(dim, dtype=torch.float32)

        return item


def collate_fn(batch):
    """배치 내 시퀀스를 패딩하여 동일 길이로 맞춤"""
    src_batch = pad_sequence([item["src"] for item in batch], batch_first=True, padding_value=0)
    tgt_batch = pad_sequence([item["tgt"] for item in batch], batch_first=True, padding_value=0)

    result = {"src": src_batch, "tgt": tgt_batch}

    if "img" in batch[0]:
        result["img"] = torch.stack([item["img"] for item in batch])

    return result


def get_dataloader(split, src_vocab, tgt_vocab, feature_type=None,
                   batch_size=64, shuffle=None, max_len=64):
    """DataLoader 생성 헬퍼"""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = Multi30KDataset(split, src_vocab, tgt_vocab, feature_type, max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=(split == "train"),
    )
    return loader


def prepare_vocabs(min_freq=2, max_vocab=10000):
    """Train 데이터로 vocab 구축 및 저장"""
    split_info = Config.SPLITS["train"]
    src_path = os.path.join(Config.DATA_DIR, split_info["en"])
    tgt_path = os.path.join(Config.DATA_DIR, split_info["de"])

    with open(src_path, "r", encoding="utf-8") as f:
        src_sents = [line.strip() for line in f if line.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_sents = [line.strip() for line in f if line.strip()]

    src_vocab = Vocabulary().build(src_sents, min_freq=min_freq, max_vocab=max_vocab)
    tgt_vocab = Vocabulary().build(tgt_sents, min_freq=min_freq, max_vocab=max_vocab)

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    src_vocab.save(os.path.join(Config.DATA_DIR, "vocab_en.json"))
    tgt_vocab.save(os.path.join(Config.DATA_DIR, "vocab_de.json"))

    print(f"Vocab built - EN: {len(src_vocab)}, DE: {len(tgt_vocab)}")
    return src_vocab, tgt_vocab
