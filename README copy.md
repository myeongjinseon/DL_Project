# Multimodal Machine Translation (MMT)

ResNet vs CLIP 이미지 피처가 번역 품질에 미치는 영향을 분석하는 프로젝트입니다.

## 프로젝트 구조

```
mmt_project/
├── configs/
│   └── config.py              # 하이퍼파라미터 & 경로 설정
├── models/
│   └── mmt_model.py           # Transformer + Image Fusion 모델
├── utils/
│   ├── vocab.py               # Vocabulary 클래스
│   ├── dataset.py             # Dataset & DataLoader
│   └── extract_features.py    # ResNet/CLIP 피처 추출기
├── data/                      # ← 여기에 데이터 배치
│   ├── train.en / train.de
│   ├── val.en / val.de
│   ├── test_2016_flickr.en / test_2016_flickr.de
│   ├── train_images.txt / val_images.txt / test_2016_flickr.txt
│   └── flickr30k-images/      # Flickr30K 이미지 폴더
├── checkpoints/               # 학습된 모델 저장
├── results/                   # 평가 결과 저장
├── train.py                   # 학습 스크립트
├── evaluate.py                # 평가 & 분석 스크립트
└── run_experiment.sh           # 전체 파이프라인 일괄 실행
```

## 데이터 준비

1. [Multi30K](https://github.com/multi30k/dataset) 에서 Task 1 텍스트 데이터 다운로드
2. [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) 이미지 다운로드
3. 위 구조에 맞게 `data/` 폴더에 배치

## 실행 방법

### 전체 파이프라인 (한 번에)
```bash
bash run_experiment.sh
```

### 단계별 실행
```bash
# 1. Vocabulary 구축
python -c "from utils.dataset import prepare_vocabs; prepare_vocabs()"

# 2. 이미지 피처 추출
python utils/extract_features.py --model resnet
python utils/extract_features.py --model clip

# 3. 모델 학습 (3가지 조건)
python train.py --mode text_only
python train.py --mode resnet
python train.py --mode clip

# 4. 평가
python evaluate.py --mode all
```

## 모델 아키텍처

- **Base**: Transformer encoder-decoder (d_model=256, 4 layers, 8 heads)
- **Fusion**: 이미지 피처를 linear projection → encoder 입력 시퀀스 앞에 prepend
- 비교 조건:
  - `text_only`: 이미지 없이 텍스트만 사용 (baseline)
  - `resnet`: ResNet-50 피처 (2048-dim)
  - `clip`: CLIP ViT-B/32 피처 (512-dim)

## 평가

- **정량**: BLEU (sacrebleu)
- **정성**: 다의어/대명사 포함 문장에서 모델별 번역 비교 (ambiguity analysis)
