"""
Multi30K 데이터셋 다운로드 및 전처리
- 텍스트 데이터: GitHub raw files에서 다운로드
- 이미지 데이터: Flickr30K 이미지 경로 매핑 (별도 다운로드 필요)
"""
import os
import json
import urllib.request
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Multi30K GitHub raw URLs (텍스트 데이터)
MULTI30K_URLS = {
    "train": {
        "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en",
        "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de",
    },
    "val": {
        "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en",
        "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de",
    },
    "test": {
        "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en",
        "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de",
    },
}

# 이미지 ID 파일 URLs
IMAGE_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/train_images.txt",
    "val": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/val_images.txt",
    "test": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/test_2016_flickr.txt",
}


def download_file(url, save_path):
    """URL에서 파일 다운로드"""
    if os.path.exists(save_path):
        print(f"  Already exists: {save_path}")
        return True
    try:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def download_multi30k():
    """Multi30K 텍스트 데이터 및 이미지 split 다운로드"""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("Downloading Multi30K text data...")
    print("=" * 50)

    for split, langs in MULTI30K_URLS.items():
        for lang, url in langs.items():
            save_path = os.path.join(DATA_DIR, f"{split}.{lang}")
            download_file(url, save_path)

    print("\nDownloading image split files...")
    for split, url in IMAGE_SPLIT_URLS.items():
        save_path = os.path.join(DATA_DIR, f"{split}_images.txt")
        download_file(url, save_path)

    print("\nDone!")


def load_text_data(split):
    """텍스트 데이터 로드 - (en_sentences, de_sentences) 리스트 반환"""
    en_path = os.path.join(DATA_DIR, f"{split}.en")
    de_path = os.path.join(DATA_DIR, f"{split}.de")

    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f.readlines()]
    with open(de_path, "r", encoding="utf-8") as f:
        de_lines = [line.strip() for line in f.readlines()]

    assert len(en_lines) == len(de_lines), \
        f"Mismatch: {len(en_lines)} EN vs {len(de_lines)} DE"

    return en_lines, de_lines


def load_image_ids(split):
    """이미지 파일명 리스트 로드"""
    img_path = os.path.join(DATA_DIR, f"{split}_images.txt")
    if not os.path.exists(img_path):
        return None
    with open(img_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def build_vocab(sentences, min_freq=2, max_vocab=10000):
    """
    단어 기반 vocabulary 구축
    (BPE 대신 단순 word-level vocab으로 시작, 나중에 BPE로 교체 가능)
    """
    counter = Counter()
    for sent in sentences:
        tokens = sent.lower().split()
        counter.update(tokens)

    # Special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}

    # Frequency 기반 vocab 추가
    for word, freq in counter.most_common(max_vocab - len(special_tokens)):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def tokenize(sentence, vocab):
    """문장을 token ID 리스트로 변환"""
    tokens = sentence.lower().split()
    ids = [vocab.get("<bos>")]
    for tok in tokens:
        ids.append(vocab.get(tok, vocab["<unk>"]))
    ids.append(vocab.get("<eos>"))
    return ids


def save_vocab(vocab, path):
    """Vocabulary를 JSON으로 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab ({len(vocab)} tokens) to {path}")


def load_vocab(path):
    """저장된 vocabulary 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data():
    """전체 데이터 준비 파이프라인"""
    # 1. 다운로드
    download_multi30k()

    # 2. 텍스트 로드
    print("\n" + "=" * 50)
    print("Loading text data...")
    print("=" * 50)

    train_en, train_de = load_text_data("train")
    val_en, val_de = load_text_data("val")
    test_en, test_de = load_text_data("test")

    print(f"  Train: {len(train_en)} pairs")
    print(f"  Val:   {len(val_en)} pairs")
    print(f"  Test:  {len(test_en)} pairs")

    # 3. Vocabulary 구축 (train 데이터 기반)
    print("\nBuilding vocabularies...")
    en_vocab = build_vocab(train_en, min_freq=2, max_vocab=10000)
    de_vocab = build_vocab(train_de, min_freq=2, max_vocab=10000)
    print(f"  EN vocab: {len(en_vocab)} tokens")
    print(f"  DE vocab: {len(de_vocab)} tokens")

    # 4. Vocabulary 저장
    save_vocab(en_vocab, os.path.join(DATA_DIR, "vocab_en.json"))
    save_vocab(de_vocab, os.path.join(DATA_DIR, "vocab_de.json"))

    # 5. 이미지 ID 로드
    print("\nLoading image IDs...")
    for split in ["train", "val", "test"]:
        img_ids = load_image_ids(split)
        if img_ids:
            print(f"  {split}: {len(img_ids)} images")
        else:
            print(f"  {split}: image list not found")

    # 6. 샘플 출력
    print("\n" + "=" * 50)
    print("Sample data (first 3 pairs):")
    print("=" * 50)
    for i in range(3):
        print(f"\n[{i}] EN: {train_en[i]}")
        print(f"    DE: {train_de[i]}")
        token_ids = tokenize(train_en[i], en_vocab)
        print(f"    EN tokens: {token_ids[:10]}...")

    return {
        "train": (train_en, train_de),
        "val": (val_en, val_de),
        "test": (test_en, test_de),
        "en_vocab": en_vocab,
        "de_vocab": de_vocab,
    }


if __name__ == "__main__":
    prepare_data()
