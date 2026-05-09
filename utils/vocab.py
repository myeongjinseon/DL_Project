"""
Vocabulary: 단어 빈도 기반 vocab 구축
"""
import json
from collections import Counter


class Vocabulary:
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.word2idx = {}
        self.idx2word = {}
        for i, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

    @property
    def pad_idx(self):
        return self.word2idx["<pad>"]

    @property
    def bos_idx(self):
        return self.word2idx["<bos>"]

    @property
    def eos_idx(self):
        return self.word2idx["<eos>"]

    @property
    def unk_idx(self):
        return self.word2idx["<unk>"]

    def __len__(self):
        return len(self.word2idx)

    def build(self, sentences, min_freq=2, max_vocab=10000):
        """문장 리스트로부터 vocab 구축"""
        counter = Counter()
        for sent in sentences:
            counter.update(sent.lower().split())

        for word, freq in counter.most_common(max_vocab - len(self.special_tokens)):
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        return self

    def encode(self, sentence, max_len=None):
        """문장 → token ID 리스트 (BOS, EOS 포함)"""
        tokens = sentence.lower().split()
        ids = [self.bos_idx]
        ids += [self.word2idx.get(t, self.unk_idx) for t in tokens]
        ids.append(self.eos_idx)
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids, skip_special=True):
        """token ID 리스트 → 문장"""
        special = {self.pad_idx, self.bos_idx, self.eos_idx}
        words = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if skip_special and idx in special:
                continue
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "special_tokens": self.special_tokens}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(data["special_tokens"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab
