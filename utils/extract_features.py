"""
이미지 피처 추출: ResNet-50 / CLIP
모든 이미지의 피처를 미리 추출하여 .npy로 저장합니다.

사용법:
    python extract_features.py --model resnet
    python extract_features.py --model clip
    python extract_features.py --model all
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config


def load_image_list(split):
    """이미지 파일명 리스트 로드"""
    img_file = Config.SPLITS[split]["img"]
    img_path = os.path.join(Config.DATA_DIR, img_file)
    with open(img_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ============================================================
# ResNet-50 Feature Extractor
# ============================================================
class ResNetExtractor:
    def __init__(self, device):
        from torchvision import models, transforms
        self.device = device

        # Pretrained ResNet-50, 마지막 FC layer 제거
        resnet = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_path):
        """단일 이미지 → 2048-dim 벡터"""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        features = self.model(img_tensor)  # (1, 2048, 1, 1)
        return features.squeeze().cpu().numpy()  # (2048,)


# ============================================================
# CLIP Feature Extractor
# ============================================================
class CLIPExtractor:
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image_path):
        """단일 이미지 → 512-dim 벡터"""
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        # L2 normalize (CLIP 표준)
        features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()  # (512,)


# ============================================================
# Main extraction pipeline
# ============================================================
def extract_features(model_type, splits=None):
    """
    지정된 모델로 모든 split의 이미지 피처 추출

    Args:
        model_type: "resnet" or "clip"
        splits: 추출할 split 리스트 (기본: 전체)
    """
    if splits is None:
        splits = ["train", "val", "test"]

    os.makedirs(Config.FEATURE_DIR, exist_ok=True)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Extractor 초기화
    print(f"\nLoading {model_type} model...")
    if model_type == "resnet":
        extractor = ResNetExtractor(device)
        feat_dim = 2048
    elif model_type == "clip":
        extractor = CLIPExtractor(device)
        feat_dim = 512
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Feature dimension: {feat_dim}")

    # 각 split에 대해 추출
    for split in splits:
        save_path = os.path.join(Config.FEATURE_DIR, f"{split}_{model_type}.npy")
        if os.path.exists(save_path):
            print(f"\n[{split}] Already extracted: {save_path}")
            continue

        print(f"\n[{split}] Extracting features...")
        image_list = load_image_list(split)
        features = np.zeros((len(image_list), feat_dim), dtype=np.float32)

        missing = 0
        for i, img_name in enumerate(tqdm(image_list, desc=f"  {split}")):
            img_path = os.path.join(Config.IMAGE_DIR, img_name)
            if os.path.exists(img_path):
                features[i] = extractor.extract(img_path)
            else:
                missing += 1

        if missing > 0:
            print(f"  WARNING: {missing}/{len(image_list)} images not found")

        np.save(save_path, features)
        print(f"  Saved: {save_path} | shape: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["resnet", "clip", "all"],
                        help="Feature extractor: resnet, clip, or all")
    args = parser.parse_args()

    if args.model == "all":
        for m in ["resnet", "clip"]:
            extract_features(m)
    else:
        extract_features(args.model)

    print("\nDone!")
