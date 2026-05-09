#!/bin/bash
# ============================================================
# MMT 전체 실험 파이프라인
# ============================================================
# 사전 준비:
#   1. data/ 폴더에 Multi30K 텍스트 파일 배치
#   2. data/flickr30k-images/ 에 이미지 배치
# ============================================================

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "Step 1: Vocabulary 구축"
echo "=============================================="
python -c "
import sys; sys.path.append('.')
from utils.dataset import prepare_vocabs
prepare_vocabs()
"

echo ""
echo "=============================================="
echo "Step 2: 이미지 피처 추출 (ResNet + CLIP)"
echo "=============================================="
python utils/extract_features.py --model all

echo ""
echo "=============================================="
echo "Step 3: Text-only 모델 학습"
echo "=============================================="
python train.py --mode text_only

echo ""
echo "=============================================="
echo "Step 4: ResNet 모델 학습"
echo "=============================================="
python train.py --mode resnet

echo ""
echo "=============================================="
echo "Step 5: CLIP 모델 학습"
echo "=============================================="
python train.py --mode clip

echo ""
echo "=============================================="
echo "Step 6: 전체 비교 평가"
echo "=============================================="
python evaluate.py --mode all

echo ""
echo "=============================================="
echo "Done! Results saved in results/"
echo "=============================================="
