"""
MMT Evaluation Script

기능:
  1. BLEU score 계산 (sacrebleu)
  2. 모델별 번역 결과 비교
  3. Ambiguity case 분석

사용법:
    python evaluate.py --mode text_only
    python evaluate.py --mode resnet
    python evaluate.py --mode clip
    python evaluate.py --mode all     # 3가지 모두 비교
"""
import os
import sys
import argparse
import json
import torch
import sacrebleu

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config
from utils.vocab import Vocabulary
from utils.dataset import Multi30KDataset
from models.mmt_model import build_model


def load_model(mode, src_vocab, tgt_vocab, device):
    """저장된 checkpoint에서 모델 로드"""
    feature_type = None if mode == "text_only" else mode
    model = build_model(len(src_vocab), len(tgt_vocab), feature_type)

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{mode}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")
    return model


def translate_dataset(model, dataset, src_vocab, tgt_vocab, device, max_len=64):
    """전체 데이터셋 번역"""
    model.eval()
    translations = []

    for i in range(len(dataset)):
        item = dataset[i]
        src = item["src"].unsqueeze(0).to(device)
        img = item["img"].unsqueeze(0).to(device) if "img" in item else None

        token_ids = model.translate(
            src, img_features=img, max_len=max_len,
            bos_idx=tgt_vocab.bos_idx, eos_idx=tgt_vocab.eos_idx,
        )
        translated = tgt_vocab.decode(token_ids)
        translations.append(translated)

    return translations


def compute_bleu(hypotheses, references):
    """sacrebleu로 BLEU 계산"""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu


def evaluate_model(mode, src_vocab, tgt_vocab, device, split="test"):
    """단일 모델 평가"""
    feature_type = None if mode == "text_only" else mode

    print(f"\n{'='*50}")
    print(f"Evaluating: {mode}")
    print(f"{'='*50}")

    # 모델 로드
    model = load_model(mode, src_vocab, tgt_vocab, device)
    if model is None:
        return None

    # 데이터 로드
    dataset = Multi30KDataset(split, src_vocab, tgt_vocab, feature_type)
    print(f"  Test samples: {len(dataset)}")

    # 번역
    print("  Translating...")
    hypotheses = translate_dataset(model, dataset, src_vocab, tgt_vocab, device)

    # Reference
    references = dataset.tgt_sentences

    # BLEU
    bleu = compute_bleu(hypotheses, references)
    print(f"\n  BLEU Score: {bleu.score:.2f}")
    print(f"  Details: {bleu}")

    return {
        "mode": mode,
        "bleu": bleu.score,
        "bleu_details": str(bleu),
        "hypotheses": hypotheses,
        "references": references,
    }


def compare_all(src_vocab, tgt_vocab, device, split="test"):
    """3가지 모드 비교 평가"""
    results = {}

    for mode in ["text_only", "resnet", "clip"]:
        result = evaluate_model(mode, src_vocab, tgt_vocab, device, split)
        if result:
            results[mode] = result

    if len(results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"{'Mode':<15} {'BLEU':>8}")
        print("-" * 25)
        for mode, r in results.items():
            print(f"{mode:<15} {r['bleu']:>8.2f}")

    return results


def analyze_ambiguity(results, src_vocab, tgt_vocab, split="test"):
    """
    Ambiguous case 분석
    다의어/대명사가 포함된 문장에서 각 모델의 번역을 비교합니다.
    """
    # 분석 대상 키워드 (다의어)
    ambiguous_words = {
        "bank": "강둑 vs 은행",
        "bat": "박쥐 vs 방망이",
        "trunk": "줄기 vs 트렁크",
        "glasses": "안경 vs 유리잔",
        "pitcher": "투수 vs 물주전자",
        "crane": "기중기 vs 학",
        "mouse": "쥐 vs 마우스",
        "spring": "봄 vs 샘",
    }

    # Test 데이터에서 ambiguous words 포함 문장 찾기
    dataset = Multi30KDataset(split, src_vocab, tgt_vocab, feature_type=None)
    src_sentences = dataset.src_sentences

    print("\n" + "=" * 60)
    print("AMBIGUITY ANALYSIS")
    print("=" * 60)

    found = 0
    for i, src in enumerate(src_sentences):
        src_lower = src.lower()
        for word, meaning in ambiguous_words.items():
            if word in src_lower.split():
                found += 1
                print(f"\n[{found}] Ambiguous word: '{word}' ({meaning})")
                print(f"  SRC: {src}")
                print(f"  REF: {dataset.tgt_sentences[i]}")
                for mode, r in results.items():
                    if i < len(r["hypotheses"]):
                        print(f"  {mode:<12}: {r['hypotheses'][i]}")

    if found == 0:
        print("  No ambiguous cases found in test set.")
        print("  Tip: Multi30K test set에서 ambiguous word가 포함된 문장을 수동으로 추가하여 분석할 수 있습니다.")

    return found


def qualitative_analysis(results, src_vocab, tgt_vocab, split="test", n_samples=10):
    """정성 평가: 모델 간 번역 차이가 큰 문장 찾기"""
    if len(results) < 2:
        return

    modes = list(results.keys())
    dataset = Multi30KDataset(split, src_vocab, tgt_vocab, feature_type=None)
    src_sentences = dataset.src_sentences

    print("\n" + "=" * 60)
    print("QUALITATIVE ANALYSIS — Sample Translations")
    print("=" * 60)

    # 랜덤 샘플
    import random
    random.seed(42)
    indices = random.sample(range(len(src_sentences)), min(n_samples, len(src_sentences)))

    for idx in indices:
        print(f"\n[{idx}]")
        print(f"  SRC: {src_sentences[idx]}")
        print(f"  REF: {dataset.tgt_sentences[idx]}")
        for mode in modes:
            if idx < len(results[mode]["hypotheses"]):
                print(f"  {mode:<12}: {results[mode]['hypotheses'][idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["text_only", "resnet", "clip", "all"])
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = Config.DEVICE

    # Vocab 로드
    src_vocab = Vocabulary.load(os.path.join(Config.DATA_DIR, "vocab_en.json"))
    tgt_vocab = Vocabulary.load(os.path.join(Config.DATA_DIR, "vocab_de.json"))
    print(f"Vocab — EN: {len(src_vocab)}, DE: {len(tgt_vocab)}")

    if args.mode == "all":
        results = compare_all(src_vocab, tgt_vocab, device, args.split)
        if results:
            analyze_ambiguity(results, src_vocab, tgt_vocab, args.split)
            qualitative_analysis(results, src_vocab, tgt_vocab, args.split)

            # 결과 저장
            os.makedirs(Config.RESULTS_DIR, exist_ok=True)
            save_data = {mode: {"bleu": r["bleu"], "bleu_details": r["bleu_details"]}
                         for mode, r in results.items()}
            with open(os.path.join(Config.RESULTS_DIR, "comparison.json"), "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"\nResults saved to {Config.RESULTS_DIR}/comparison.json")
    else:
        evaluate_model(args.mode, src_vocab, tgt_vocab, device, args.split)
