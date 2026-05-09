"""
MMT Project Configuration
"""
import os
import torch

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "flickr30k-images")
    FEATURE_DIR = os.path.join(DATA_DIR, "features")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    SPLITS = {
        "train": {"en": "train.en", "de": "train.de", "img": "train_images.txt"},
        "val":   {"en": "val.en",   "de": "val.de",   "img": "val_images.txt"},
        "test":  {"en": "test_2016_flickr.en", "de": "test_2016_flickr.de", "img": "test_2016_flickr.txt"},
    }

    SRC_LANG = "en"
    TGT_LANG = "de"
    MAX_SEQ_LEN = 64
    MIN_FREQ = 2

    D_MODEL = 256
    N_HEADS = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    D_FF = 512
    DROPOUT = 0.3

    IMG_FEATURE_DIM = {"resnet": 2048, "clip": 512}
    IMG_PROJ_DIM = 256

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 4000
    MAX_EPOCHS = 30
    PATIENCE = 5
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
