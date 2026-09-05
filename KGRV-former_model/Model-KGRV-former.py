
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm, trange

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

try:
    from torch_geometric.nn import GATv2Conv as _GraphConv
except Exception:
    try:
        from torch_geometric.nn import GATConv as _GraphConv
    except Exception:
        _GraphConv = None



# CONFIG

class Config:
    DATA_PATH = "/root/.ssh/Jupyter_SVS-BERT/Jupyter_SVS-BERT/newextendeddataset.csv"
    MODEL_NAME = "bert-base-uncased"
    OUTPUT_DIR = "./outputs/va_g1_fgm_v3_step3_1_dynamic_fusion"
    MODE = "full"  # quick | full

    MAX_LEN = 128
    LABEL_ORDER = [
        "self_direction",
        "stimulation",
        "hedonism",
        "achievement",
        "power",
        "security",
        "conformity",
        "tradition",
        "benevolence",
        "universalism",
    ]
    DISPLAY_LABELS = [x.replace("_", " ") for x in LABEL_ORDER]
    NUM_LABELS = len(LABEL_ORDER)

    BATCH_SIZE = 8
    EPOCHS = 8
    LR_CANDIDATES = [3e-5, 4e-5]
    PSO_SWARM_SIZE = 4
    PSO_ITERS = 3
    PSO_W = 0.72
    PSO_C1 = 1.49
    PSO_C2 = 1.49
    PSO_LR_MIN = 1e-5
    PSO_LR_MAX = 5e-5
    PSO_WEIGHT_DECAY_MIN = 0.0
    PSO_WEIGHT_DECAY_MAX = 0.03
    PSO_DROPOUT_MIN = 0.05
    PSO_DROPOUT_MAX = 0.20
    PSO_GAT_DROPOUT_MIN = 0.10
    PSO_GAT_DROPOUT_MAX = 0.30
    PSO_CONTRASTIVE_WEIGHT_MIN = 0.0
    PSO_CONTRASTIVE_WEIGHT_MAX = 0.08
    PSO_VALUE_REG_WEIGHT_MIN = 0.0
    PSO_VALUE_REG_WEIGHT_MAX = 0.05
    PSO_GRAPH_MIX_INIT_MIN = 0.05
    PSO_GRAPH_MIX_INIT_MAX = 0.30
    PSO_FGM_EPS_MIN = 0.15
    PSO_FGM_EPS_MAX = 0.60
    PSO_LOGIT_SCALE_INIT_MIN = 6.0
    PSO_LOGIT_SCALE_INIT_MAX = 14.0
    WEIGHT_DECAY = 0.010984
    WARMUP_RATIO = 0.10
    MAX_GRAD_NORM = 1.0

    TEST_SIZE = 0.15
    LR_SEARCH_VAL_SIZE = 0.20
    QUICK_MODEL_SELECTION_VAL_SIZE = 0.10
    FULL_MODEL_SELECTION_VAL_SIZE = 0.15
    CV_FOLDS = 5

    DROPOUT = 0.084418
    GRAPH_DIM = 128
    GAT_HEADS = 2
    GAT_DROPOUT = 0.270805
    EDGE_ATTR_DIM = 8
    LOGIT_SCALE_INIT = 11.707882

    CONTRASTIVE_WEIGHT = 0.046682
    CONTRASTIVE_TEMP = 0.10
    CONTRASTIVE_WARMUP_EPOCHS = 3

    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 1.5

    VALUE_REG_WEIGHT = 0.024336
    VALUE_REG_WARMUP_EPOCHS = 2
    POS_MARGIN = 0.35
    NEG_MARGIN = -0.10

    GRAPH_MIX_INIT = 0.117678
    GATE_BIAS_INIT = -2.00

    USE_WEIGHTED_SAMPLER = True
    SYNTH_ENABLE = True
    SYNTH_TRIGGER_RATIO = 0.30
    SYNTH_MAX_MULTIPLIER = 1.25
    SYNTH_DELETE_P = 0.04
    SYNTH_SWAP_OPS = 1
    SYNTH_MAX_TOKENS = 96

    USE_FGM = True
    FGM_EPS = 0.204652
    FGM_EMB_NAME = "word_embeddings"

    FINAL_EPOCH_BUFFER = 0
    FINAL_MIN_EPOCHS = 8

    # Refined graph/prototype experiment settings
    # Graph topology: remove distance-2 message passing and explicitly block
    # empirically confused pairs so the GAT does not mix their evidence.
    USE_TWO_STEP_GRAPH_EDGES = False
    BLOCK_HARD_CONFUSION_GRAPH_EDGES = True
    BLOCKED_GRAPH_PAIRS = {
        (6, 8),  # conformity ↔ benevolence
        (6, 2),  # conformity ↔ hedonism
        (8, 2),  # benevolence ↔ hedonism
    }

    # Prototype regularizer: softer target-similarity matrix. This keeps useful
    # Schwartz-style structure but avoids forcing ambiguous classes to collapse.
    USE_TARGET_STRUCTURE_REG = True
    STRUCTURE_DEFAULT_NEIGHBOR_TARGET = 0.15
    STRUCTURE_DEFAULT_TWO_STEP_TARGET = 0.00
    STRUCTURE_DEFAULT_OPPOSITE_TARGET = -0.25
    STRUCTURE_OTHER_TARGET = 0.0
    STRUCTURE_HARD_TARGETS = {
        (6, 8): -0.05,  # conformity vs benevolence: mild separation
        (6, 2): -0.15,  # conformity vs hedonism: stronger separation
        (8, 2):  0.00,  # benevolence vs hedonism: neutral
        (6, 5):  0.15,  # conformity and security: related but not collapsed
        (6, 7):  0.15,  # conformity and tradition: related but not collapsed
    }

    # Direct prototype-classifier branch. The prototypes are used directly
    # in the final logits, not only as weak additive node priors.
    PROTO_MIX_INIT = 0.20

    # These flags are switched per output script below.
    USE_SEMANTIC_LABEL_PROTOTYPES = True
    USE_LABELWISE_CROSS_ATTENTION = True
    USE_HIGHER_ORDER_HEAD = True

    # Schwartz relation-biased label
    # Graph Transformer. This replaces local GAT message passing with global
    # relation-aware label self-attention. The 10 Schwartz values can attend to
    # all other values, but every pair receives a learned bias according to its
    # theory/empirical relation type: self, compatible neighbor, same higher-
    # order region, opposing value, empirical confusion, or neutral.
    USE_RELATION_GRAPH_TRANSFORMER = True
    RELATION_NUM_TYPES = 6
    RELATION_NUM_HEADS = 4
    RELATION_NUM_LAYERS = 2
    RELATION_DROPOUT = 0.15
    REL_SELF = 0
    REL_COMPATIBLE_NEIGHBOR = 1
    REL_SAME_HIGHER_ORDER = 2
    REL_OPPOSING = 3
    REL_EMPIRICAL_CONFUSION = 4
    REL_NEUTRAL = 5

    # Semantic label-description prototypes.
    # The descriptions are encoded once by the frozen/current BERT backbone at
    # model construction and projected into graph space. They form a theory-aware
    # semantic anchor for abstract Schwartz values.
    SEMANTIC_PROTO_MIX_INIT = 0.35
    SEMANTIC_ANCHOR_WEIGHT = 0.01
    LABEL_DESCRIPTIONS = [
        "self direction independent thought creativity curiosity freedom choosing own goals",
        "stimulation excitement novelty challenge varied life adventure",
        "hedonism pleasure enjoyment fun comfort satisfaction gratification",
        "achievement success ambition competence capability recognition accomplishment",
        "power authority social status control dominance influence resources",
        "security safety stability order protection privacy reliability social harmony",
        "conformity obedience rules compliance discipline social norms politeness avoiding violation",
        "tradition respect customs religion culture heritage commitment humility",
        "benevolence helping caring loyalty support responsibility welfare of close others family friends",
        "universalism equality justice tolerance broad-mindedness nature environment social welfare",
    ]

    # Label-wise text-value cross-attention classifier branch.
    # This branch converts each label-specific token evidence vector into its own
    # class logit, so conformity can attend to rule/compliance evidence while
    # benevolence attends to help/care evidence.
    LABELWISE_MIX_INIT = 0.20

    # sample-adaptive dynamic logit fusion
    # Instead of using only global scalar mix weights, this module predicts
    # per-sample weights over base, graph, prototype, label-wise, and higher-order
    # logit channels.
    USE_SAMPLE_DYNAMIC_FUSION = True
    DYNAMIC_FUSION_DROPOUT = 0.10
    DYNAMIC_FUSION_CHANNELS = 5

    # Higher-order Schwartz auxiliary supervision.
    # Standard four Schwartz higher-order regions are used. Hedonism is treated
    # as a soft boundary value with 0.5 membership in openness_to_change and
    # 0.5 membership in self_enhancement, rather than as a non-standard fifth
    # class.
    HIGHER_LABEL_ORDER = [
        "openness_to_change",
        "self_enhancement",
        "conservation",
        "self_transcendence",
    ]
    NUM_HIGHER_LABELS = 4
    FINE_TO_HIGHER_SOFT = [
        [1.0, 0.0, 0.0, 0.0],  # self_direction -> openness_to_change
        [1.0, 0.0, 0.0, 0.0],  # stimulation -> openness_to_change
        [0.5, 0.5, 0.0, 0.0],  # hedonism -> boundary: openness + self_enhancement
        [0.0, 1.0, 0.0, 0.0],  # achievement -> self_enhancement
        [0.0, 1.0, 0.0, 0.0],  # power -> self_enhancement
        [0.0, 0.0, 1.0, 0.0],  # security -> conservation
        [0.0, 0.0, 1.0, 0.0],  # conformity -> conservation
        [0.0, 0.0, 1.0, 0.0],  # tradition -> conservation
        [0.0, 0.0, 0.0, 1.0],  # benevolence -> self_transcendence
        [0.0, 0.0, 0.0, 1.0],  # universalism -> self_transcendence
    ]
    HIGHER_AUX_WEIGHT = 0.12
    HIGHER_LOGIT_MIX = 0.08

    # The strong confusion-margin loss and hard-triad sampler hurt
    # global generalization. So they are disabled by default in this refined run.
    USE_CONFUSION_MARGIN = False
    CONF_MARGIN = 0.50
    CONF_LOSS_WEIGHT = 0.0
    CONFUSION_HARD_NEGATIVES = {
        6: [8, 2],
        8: [6],
        2: [6],
    }

    USE_HARD_TRIAD_SAMPLER = False
    HARD_TRIAD_CLASSES = [2, 6, 8]
    HARD_TRIAD_BOOST = 1.0
    HARD_CONFORMITY_BOOST = 1.0

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed PSO-selected hyperparameters
    # Selected from the previous KGRV-former PSO run, this final script skips PSO
    # search and trains directly with this configuration.
    BEST_PSO_CONFIG = {
        "lr": 5e-05,
        "weight_decay": 0.010984,
        "dropout": 0.084418,
        "gat_dropout": 0.270805,
        "contrastive_weight": 0.046682,
        "value_reg_weight": 0.024336,
        "graph_mix_init": 0.117678,
        "fgm_eps": 0.204652,
        "logit_scale_init": 11.707882,
    }


# UTILITIES

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(obj: dict, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def clean_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "none"
    x = str(x)
    x = re.sub(r"\s+", " ", x)
    x = x.strip()
    return x if x else "none"


def normalize_label_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str | Path, title: str):
    if plt is None or sns is None:
        return
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_learning_curves(history: dict, out_path: str | Path):
    if plt is None:
        return
    epochs = np.arange(1, len(history["train"]) + 1)
    train_loss = [x["loss"] for x in history["train"]]
    val_loss = [x["loss"] for x in history["val"]]
    train_f1 = [x["macro_f1"] for x in history["train"]]
    val_f1 = [x["macro_f1"] for x in history["val"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="s", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_f1, marker="o", label="Train Macro-F1")
    axes[1].plot(epochs, val_f1, marker="s", label="Val Macro-F1")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# TEXT AUGMENTATION

def random_delete(tokens: List[str], rng: random.Random, p: float) -> List[str]:
    if len(tokens) <= 4:
        return tokens[:]
    kept = [tok for tok in tokens if rng.random() > p]
    return kept if kept else tokens[:]


def random_swap(tokens: List[str], rng: random.Random, n_swaps: int) -> List[str]:
    toks = tokens[:]
    if len(toks) < 3:
        return toks
    for _ in range(n_swaps):
        i, j = rng.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks


def span_mix(tokens_a: List[str], tokens_b: List[str], rng: random.Random) -> List[str]:
    if not tokens_a:
        return tokens_b[:]
    if not tokens_b:
        return tokens_a[:]
    cut_a = 1 if len(tokens_a) < 3 else rng.randint(1, max(1, len(tokens_a) - 1))
    cut_b = 1 if len(tokens_b) < 3 else rng.randint(1, max(1, len(tokens_b) - 1))
    mixed = tokens_a[:cut_a] + tokens_b[cut_b:]
    return mixed if mixed else tokens_a[:]


def synthesize_text(text_a: str, text_b: str, rng: random.Random) -> str:
    toks_a = clean_text(text_a).split()
    toks_b = clean_text(text_b).split()
    op = rng.choice(["mix", "mix_swap", "delete_swap"])
    if op == "mix":
        toks = span_mix(toks_a, toks_b, rng)
    elif op == "mix_swap":
        toks = random_swap(span_mix(toks_a, toks_b, rng), rng, Config.SYNTH_SWAP_OPS)
    else:
        base = toks_a if rng.random() < 0.5 else toks_b
        toks = random_delete(base, rng, Config.SYNTH_DELETE_P)
        toks = random_swap(toks, rng, Config.SYNTH_SWAP_OPS)
    toks = toks[: Config.SYNTH_MAX_TOKENS]
    text = " ".join(toks).strip()
    return text if text else clean_text(text_a)


def augment_training_texts(texts: np.ndarray, labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict]]:
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    if not Config.SYNTH_ENABLE:
        return texts, labels, {"mode": "none", "classes": {}}

    counts = Counter(labels.tolist())
    positive_counts = [v for v in counts.values() if v > 0]
    if not positive_counts:
        return texts, labels, {"mode": "none", "classes": {}}

    median_count = int(np.median(positive_counts))
    trigger_threshold = max(2, int(round(Config.SYNTH_TRIGGER_RATIO * median_count)))

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for t, y in zip(texts, labels):
        grouped[int(y)].append(clean_text(t))

    out_texts = list(map(clean_text, texts.tolist()))
    out_labels = labels.tolist()
    summary = {
        "mode": "hybrid_weighted_sampler_plus_synthetic_text",
        "median_count": int(median_count),
        "trigger_threshold": int(trigger_threshold),
        "classes": {},
    }

    for cls_idx in range(Config.NUM_LABELS):
        original = counts.get(cls_idx, 0)
        target = original
        synth_needed = 0

        if original > 0 and original < trigger_threshold:
            target = max(original, int(math.ceil(original * Config.SYNTH_MAX_MULTIPLIER)))
            target = min(target, trigger_threshold)
            synth_needed = max(0, target - original)

        synthetic = []
        if synth_needed > 0:
            pool = grouped[cls_idx]
            for _ in range(synth_needed):
                if len(pool) == 1:
                    ta = tb = pool[0]
                else:
                    ta, tb = rng.sample(pool, 2)
                synthetic.append(synthesize_text(ta, tb, rng))

        out_texts.extend(synthetic)
        out_labels.extend([cls_idx] * len(synthetic))
        summary["classes"][Config.LABEL_ORDER[cls_idx]] = {
            "original": int(original),
            "target": int(target),
            "synthetic_added": int(len(synthetic)),
        }

    combined = list(zip(out_texts, out_labels))
    rng.shuffle(combined)
    out_texts, out_labels = zip(*combined)
    return np.asarray(out_texts), np.asarray(out_labels), summary



# DATASET & SAMPLER

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = [clean_text(x) for x in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=Config.NUM_LABELS)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = np.asarray([weights[y] for y in labels], dtype=np.float64)

    if getattr(Config, "USE_HARD_TRIAD_SAMPLER", False):
        triad_classes = set(int(x) for x in Config.HARD_TRIAD_CLASSES)
        for i, y in enumerate(labels):
            if int(y) == 6:  # conformity
                sample_weights[i] *= float(Config.HARD_CONFORMITY_BOOST)
            elif int(y) in triad_classes:
                sample_weights[i] *= float(Config.HARD_TRIAD_BOOST)

    sample_weights = sample_weights / max(sample_weights.mean(), 1e-12)
    return WeightedRandomSampler(torch.from_numpy(sample_weights), len(labels), replacement=True)


# LOSS FUNCTIONS

class SupConLoss(nn.Module):
    def __init__(self, temp: float = 0.1):
        super().__init__()
        self.temp = temp

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1) / self.temp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        same.fill_diagonal_(0.0)
        denom_mask = 1.0 - torch.eye(sim.size(0), device=sim.device)
        exp_sim = torch.exp(sim) * denom_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(same * log_prob).sum(dim=1) / (same.sum(dim=1) + 1e-8)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


def confusion_margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not getattr(Config, "USE_CONFUSION_MARGIN", False):
        return logits.new_tensor(0.0)

    terms = []
    margin = float(Config.CONF_MARGIN)
    for true_cls, neg_classes in Config.CONFUSION_HARD_NEGATIVES.items():
        mask = labels == int(true_cls)
        if not torch.any(mask):
            continue
        true_logits = logits[mask, int(true_cls)]
        for neg_cls in neg_classes:
            neg_logits = logits[mask, int(neg_cls)]
            terms.append(F.relu(margin - (true_logits - neg_logits)).pow(2).mean())

    if not terms:
        return logits.new_tensor(0.0)
    return torch.stack(terms).mean()


# GRAPH CONSTRUCTION

def _canonical_pair(a: int, b: int) -> Tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def build_sparse_schwartz_graph(num_labels: int = 10):
    blocked = set()
    if getattr(Config, "BLOCK_HARD_CONFUSION_GRAPH_EDGES", True):
        blocked = {_canonical_pair(a, b) for (a, b) in Config.BLOCKED_GRAPH_PAIRS}

    edges = []
    seen = set()

    for i in range(num_labels):
        candidates = [(i, 0)]  # self evidence

        # direct neighbours only
        candidates.append(((i + 1) % num_labels, 1))
        candidates.append(((i - 1) % num_labels, 1))

        # optional distance-2 edges; disabled by default for this experiment
        if getattr(Config, "USE_TWO_STEP_GRAPH_EDGES", False):
            candidates.append(((i + 2) % num_labels, 2))
            candidates.append(((i - 2) % num_labels, 2))

        # opposite relation
        candidates.append(((i + num_labels // 2) % num_labels, 3))

        for j, et in candidates:
            if i != j and _canonical_pair(i, j) in blocked:
                continue
            key = (i, j, et)
            if key not in seen:
                seen.add(key)
                edges.append((i, j, et))

    src = torch.tensor([x[0] for x in edges], dtype=torch.long)
    dst = torch.tensor([x[1] for x in edges], dtype=torch.long)
    etype = torch.tensor([x[2] for x in edges], dtype=torch.long)
    return torch.stack([src, dst], dim=0), etype

def build_structure_target_pairs(num_labels: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    hard_targets = {}
    for (a, b), target in Config.STRUCTURE_HARD_TARGETS.items():
        key = tuple(sorted((int(a), int(b))))
        hard_targets[key] = float(target)

    pairs, targets = [], []
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            key = (i, j)
            if key in hard_targets:
                target = hard_targets[key]
            else:
                dist = min(abs(i - j), num_labels - abs(i - j))
                if dist == 1:
                    target = Config.STRUCTURE_DEFAULT_NEIGHBOR_TARGET
                elif dist == 2:
                    target = Config.STRUCTURE_DEFAULT_TWO_STEP_TARGET
                elif dist == num_labels // 2:
                    target = Config.STRUCTURE_DEFAULT_OPPOSITE_TARGET
                else:
                    target = Config.STRUCTURE_OTHER_TARGET
            pairs.append((i, j))
            targets.append(float(target))

    return torch.tensor(pairs, dtype=torch.long), torch.tensor(targets, dtype=torch.float32)


def batch_graph(edge_index: torch.Tensor, edge_type: torch.Tensor, batch_size: int, n_nodes: int, device: torch.device):
    offsets = (torch.arange(batch_size, device=device) * n_nodes).view(batch_size, 1, 1)
    batched_edge_index = (edge_index.to(device).unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    batched_edge_type = edge_type.to(device).repeat(batch_size)
    return batched_edge_index, batched_edge_type


def _dominant_higher_order_index(fine_idx: int) -> int:
    row = np.asarray(Config.FINE_TO_HIGHER_SOFT[int(fine_idx)], dtype=np.float32)
    return int(row.argmax())


def build_schwartz_relation_type_matrix(num_labels: int = 10) -> torch.Tensor:
    rel = torch.full((num_labels, num_labels), int(Config.REL_NEUTRAL), dtype=torch.long)
    hard_confusion = {_canonical_pair(a, b) for (a, b) in Config.BLOCKED_GRAPH_PAIRS}

    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                rel[i, j] = int(Config.REL_SELF)
                continue

            key = _canonical_pair(i, j)
            dist = min(abs(i - j), num_labels - abs(i - j))
            same_higher = _dominant_higher_order_index(i) == _dominant_higher_order_index(j)

            if key in hard_confusion:
                rel[i, j] = int(Config.REL_EMPIRICAL_CONFUSION)
            elif dist == num_labels // 2:
                rel[i, j] = int(Config.REL_OPPOSING)
            elif same_higher:
                rel[i, j] = int(Config.REL_SAME_HIGHER_ORDER)
            elif dist == 1:
                rel[i, j] = int(Config.REL_COMPATIBLE_NEIGHBOR)
            else:
                rel[i, j] = int(Config.REL_NEUTRAL)
    return rel


class RelationBiasedLabelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_relation_types: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"GRAPH_DIM={dim} must be divisible by RELATION_NUM_HEADS={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Embedding(num_relation_types, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.rel_bias.weight)

    def forward(self, x: torch.Tensor, relation_types: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        q = self.q_proj(x).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # relation bias: [N, N, H] -> [1, H, N, N]
        rb = self.rel_bias(relation_types).permute(2, 0, 1).unsqueeze(0)
        scores = scores + rb

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n_nodes, self.dim)
        return self.out_dropout(self.out_proj(ctx))


class RelationBiasedLabelTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_relation_types: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelationBiasedLabelAttention(dim, num_heads, num_relation_types, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, relation_types: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), relation_types)
        x = x + self.ffn(self.norm2(x))
        return x


class RelationBiasedLabelTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, num_relation_types: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationBiasedLabelTransformerLayer(dim, num_heads, num_relation_types, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, relation_types: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, relation_types)
        return self.final_norm(x)

# FGM ADVERSARIAL TRAINING

class FGM:
    def __init__(self, model: nn.Module, emb_name: str = "word_embeddings", epsilon: float = 0.35):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                grad = param.grad.detach()
                norm = torch.norm(grad)
                if torch.isfinite(norm) and norm > 0:
                    self.backup[name] = param.data.clone()
                    r_at = self.epsilon * grad / (norm + 1e-12)
                    param.data.add_(r_at)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# MODEL

class ValueAwareGATBERT(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.last4_weights = nn.Parameter(torch.ones(4))

        self.review_proj = nn.Linear(hidden_size, Config.GRAPH_DIM)

        self.value_queries = nn.Parameter(torch.randn(Config.NUM_LABELS, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.global_prototypes = nn.Parameter(torch.randn(Config.NUM_LABELS, Config.GRAPH_DIM) * 0.02)
        self.semantic_proto_proj = nn.Linear(hidden_size, Config.GRAPH_DIM)
        sem_mix = float(getattr(Config, "SEMANTIC_PROTO_MIX_INIT", 0.35))
        sem_mix = min(max(sem_mix, 1e-4), 1.0 - 1e-4)
        self.semantic_proto_mix_logit = nn.Parameter(torch.tensor(math.log(sem_mix / (1.0 - sem_mix)), dtype=torch.float))
        label_desc_hidden = self._encode_label_descriptions(hidden_size)
        self.register_buffer("label_desc_hidden", label_desc_hidden)

        self.node_text = nn.Linear(hidden_size, Config.GRAPH_DIM)
        self.node_review = nn.Linear(Config.GRAPH_DIM, Config.GRAPH_DIM)

        self.label_transformer = RelationBiasedLabelTransformer(
            dim=Config.GRAPH_DIM,
            num_heads=Config.RELATION_NUM_HEADS,
            num_layers=Config.RELATION_NUM_LAYERS,
            num_relation_types=Config.RELATION_NUM_TYPES,
            dropout=Config.RELATION_DROPOUT,
        )

        self.node_gate = nn.Linear(Config.GRAPH_DIM * 3, 1)
        nn.init.constant_(self.node_gate.bias, Config.GATE_BIAS_INIT)

        self.node_pool = nn.Linear(Config.GRAPH_DIM * 2, 1)
        self.fuse = nn.Linear(Config.GRAPH_DIM * 2, Config.GRAPH_DIM)

        self.base_review_head = nn.Linear(Config.GRAPH_DIM, Config.GRAPH_DIM)
        self.graph_review_head = nn.Linear(Config.GRAPH_DIM, Config.GRAPH_DIM)

        self.labelwise_text_proj = nn.Linear(hidden_size, Config.GRAPH_DIM)
        self.labelwise_scorer = nn.Sequential(
            nn.Linear(Config.GRAPH_DIM * 3, Config.GRAPH_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.GRAPH_DIM, 1),
        )
        lw_mix = float(getattr(Config, "LABELWISE_MIX_INIT", 0.20))
        lw_mix = min(max(lw_mix, 1e-4), 1.0 - 1e-4)
        self.labelwise_mix_logit = nn.Parameter(torch.tensor(math.log(lw_mix / (1.0 - lw_mix)), dtype=torch.float))

        self.dynamic_logit_fusion = nn.Sequential(
            nn.Linear(Config.GRAPH_DIM * 3, Config.GRAPH_DIM),
            nn.GELU(),
            nn.Dropout(float(getattr(Config, "DYNAMIC_FUSION_DROPOUT", Config.DROPOUT))),
            nn.Linear(Config.GRAPH_DIM, int(getattr(Config, "DYNAMIC_FUSION_CHANNELS", 5))),
        )

        self.higher_head = nn.Linear(Config.GRAPH_DIM, Config.NUM_HIGHER_LABELS)
        self.higher_to_fine = nn.Linear(Config.NUM_HIGHER_LABELS, Config.NUM_LABELS, bias=False)
        higher_soft = torch.tensor(Config.FINE_TO_HIGHER_SOFT, dtype=torch.float32)
        with torch.no_grad():
            self.higher_to_fine.weight.copy_(higher_soft)
        self.register_buffer("fine_to_higher_soft", higher_soft)

        mix_logit = math.log(Config.GRAPH_MIX_INIT / (1.0 - Config.GRAPH_MIX_INIT))
        self.graph_mix_logit = nn.Parameter(torch.tensor(mix_logit, dtype=torch.float))

        proto_mix = float(getattr(Config, "PROTO_MIX_INIT", 0.20))
        proto_mix = min(max(proto_mix, 1e-4), 1.0 - 1e-4)
        proto_mix_logit = math.log(proto_mix / (1.0 - proto_mix))
        self.proto_mix_logit = nn.Parameter(torch.tensor(proto_mix_logit, dtype=torch.float))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(Config.LOGIT_SCALE_INIT), dtype=torch.float))
        self.logit_bias = nn.Parameter(torch.zeros(Config.NUM_LABELS))

        self.proj = nn.Linear(Config.GRAPH_DIM, 128)
        self.supcon = SupConLoss(Config.CONTRASTIVE_TEMP)
        self.focal = FocalLoss(Config.FOCAL_ALPHA, Config.FOCAL_GAMMA)

        relation_types = build_schwartz_relation_type_matrix(Config.NUM_LABELS)
        self.register_buffer("relation_types", relation_types)

        struct_pairs, struct_targets = build_structure_target_pairs(Config.NUM_LABELS)
        self.register_buffer("struct_pairs", struct_pairs)
        self.register_buffer("struct_targets", struct_targets)

    def _encode_label_descriptions(self, hidden_size: int) -> torch.Tensor:
        try:
            tok = BertTokenizerFast.from_pretrained(Config.MODEL_NAME)
            enc = tok(
                list(Config.LABEL_DESCRIPTIONS),
                truncation=True,
                padding="max_length",
                max_length=48,
                return_tensors="pt",
            )
            device = next(self.bert.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            was_training = self.bert.training
            self.bert.eval()
            with torch.no_grad():
                out = self.bert(**enc, return_dict=True)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                desc = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            if was_training:
                self.bert.train()
            return desc.detach().cpu()
        except Exception:
            g = torch.Generator().manual_seed(Config.SEED)
            return torch.randn(Config.NUM_LABELS, hidden_size, generator=g) * 0.02

    def semantic_prototypes(self) -> torch.Tensor:
        return self.semantic_proto_proj(self.label_desc_hidden.to(self.global_prototypes.device))

    def effective_prototypes(self) -> torch.Tensor:
        learned = self.global_prototypes
        if not getattr(Config, "USE_SEMANTIC_LABEL_PROTOTYPES", False):
            return learned
        semantic = self.semantic_prototypes()
        rho = torch.sigmoid(self.semantic_proto_mix_logit)
        return (1.0 - rho) * learned + rho * semantic

    def mix_last4_tokens(self, hidden_states):
        stack = torch.stack(hidden_states[-4:], dim=1)
        w = torch.softmax(self.last4_weights, dim=0).view(1, 4, 1, 1)
        return (stack * w).sum(dim=1)

    def masked_mean(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

    def token_condition_nodes(self, tokens: torch.Tensor, review: torch.Tensor, attention_mask: torch.Tensor):
        q = self.q_proj(self.value_queries)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        scale = math.sqrt(k.size(-1))
        scores = torch.einsum("nh,bth->bnt", q, k) / scale
        scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e4)
        attn = torch.softmax(scores, dim=-1)
        evidence = torch.einsum("bnt,bth->bnh", attn, v)

        proto = self.effective_prototypes()
        node0 = self.node_text(evidence) + self.node_review(review).unsqueeze(1) + proto.unsqueeze(0)
        node0 = F.layer_norm(node0, (Config.GRAPH_DIM,))
        return node0, attn, evidence

    def value_regularizer(self) -> torch.Tensor:
      
        if not getattr(Config, "USE_TARGET_STRUCTURE_REG", True):
            return self.global_prototypes.new_tensor(0.0)

        p = F.normalize(self.effective_prototypes(), dim=-1)
        sim = p @ p.t()
        pair_sim = sim[self.struct_pairs[:, 0], self.struct_pairs[:, 1]]
        targets = self.struct_targets.to(pair_sim.device)
        return F.mse_loss(pair_sim, targets)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, epoch_idx: int = 0):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        tokens = self.mix_last4_tokens(out.hidden_states)
        review = self.review_proj(self.dropout(self.masked_mean(tokens, attention_mask)))
        review = F.normalize(review, dim=-1)

        node0, token_attn, label_evidence = self.token_condition_nodes(tokens, review, attention_mask)

        batch_size = review.size(0)
        node1 = self.label_transformer(node0, self.relation_types.to(node0.device))

        review_rep = review.unsqueeze(1).expand_as(node0)
        gate_in = torch.cat([node0, node1, review_rep], dim=-1)
        gate = torch.sigmoid(self.node_gate(gate_in))
        node_tilde = node0 + gate * (node1 - node0)

        pool_in = torch.cat([node_tilde, review_rep], dim=-1)
        pool_scores = self.node_pool(pool_in).squeeze(-1)
        pool_weights = torch.softmax(pool_scores, dim=-1)
        graph_summary = torch.einsum("bn,bnd->bd", pool_weights, node_tilde)

        fused = self.fuse(torch.cat([review, graph_summary], dim=-1))
        fused = F.normalize(fused, dim=-1)

        base_review = F.normalize(self.base_review_head(review), dim=-1)
        graph_review = F.normalize(self.graph_review_head(fused), dim=-1)

        base_nodes = F.normalize(node0, dim=-1)
        graph_nodes = F.normalize(node_tilde, dim=-1)

        base_logits = torch.einsum("bd,bnd->bn", base_review, base_nodes)
        graph_logits = torch.einsum("bd,bnd->bn", graph_review, graph_nodes)

        proto_nodes = F.normalize(self.effective_prototypes(), dim=-1)
        proto_logits = torch.einsum("bd,nd->bn", graph_review, proto_nodes)

        labelwise_logits = torch.zeros_like(base_logits)
        if getattr(Config, "USE_LABELWISE_CROSS_ATTENTION", False):
            label_text = F.normalize(self.labelwise_text_proj(self.dropout(label_evidence)), dim=-1)
            lw_in = torch.cat([label_text, graph_nodes, review_rep], dim=-1)
            labelwise_logits = self.labelwise_scorer(lw_in).squeeze(-1)

        higher_logits = self.higher_head(fused)
        higher_prior_logits = torch.zeros_like(base_logits)
        if getattr(Config, "USE_HIGHER_ORDER_HEAD", False):
            higher_prior_logits = self.higher_to_fine(higher_logits)

        if getattr(Config, "USE_SAMPLE_DYNAMIC_FUSION", False):
            fusion_context = torch.cat([review, graph_summary, fused], dim=-1)
            dynamic_fusion_weights = torch.softmax(self.dynamic_logit_fusion(fusion_context), dim=-1)
            channels = [
                base_logits,
                graph_logits,
                proto_logits,
                labelwise_logits,
                higher_prior_logits,
            ]
            logits = sum(dynamic_fusion_weights[:, i:i + 1] * channels[i] for i in range(len(channels)))
            beta = dynamic_fusion_weights[:, 1].mean().detach()
            gamma = dynamic_fusion_weights[:, 2].mean().detach()
            eta = dynamic_fusion_weights[:, 3].mean().detach()
        else:
            dynamic_fusion_weights = None
            beta = torch.sigmoid(self.graph_mix_logit)
            gamma = torch.sigmoid(self.proto_mix_logit)
            eta = torch.sigmoid(self.labelwise_mix_logit)
            node_logits = (1.0 - beta) * base_logits + beta * graph_logits
            logits = (1.0 - gamma) * node_logits + gamma * proto_logits
            if getattr(Config, "USE_LABELWISE_CROSS_ATTENTION", False):
                logits = (1.0 - eta) * logits + eta * labelwise_logits
            if getattr(Config, "USE_HIGHER_ORDER_HEAD", False):
                logits = logits + float(Config.HIGHER_LOGIT_MIX) * higher_prior_logits

        logits = logits * self.logit_scale.exp() + self.logit_bias

        losses = {}
        if labels is not None:
            losses["focal"] = self.focal(logits, labels)
            losses["confusion_margin"] = confusion_margin_loss(logits, labels)
            z = F.normalize(self.proj(fused), dim=-1)
            losses["supcon"] = self.supcon(z, labels)
            if epoch_idx + 1 <= Config.VALUE_REG_WARMUP_EPOCHS:
                losses["value_reg"] = torch.tensor(0.0, device=logits.device)
            else:
                losses["value_reg"] = self.value_regularizer()

            if getattr(Config, "USE_SEMANTIC_LABEL_PROTOTYPES", False):
                learned_p = F.normalize(self.global_prototypes, dim=-1)
                semantic_p = F.normalize(self.semantic_prototypes(), dim=-1)
                losses["semantic_anchor"] = F.mse_loss(learned_p, semantic_p)
            else:
                losses["semantic_anchor"] = torch.tensor(0.0, device=logits.device)

            if getattr(Config, "USE_HIGHER_ORDER_HEAD", False):
                higher_targets = self.fine_to_higher_soft[labels].to(higher_logits.device)
                higher_log_probs = F.log_softmax(higher_logits, dim=-1)
                losses["higher_aux"] = -(higher_targets * higher_log_probs).sum(dim=-1).mean()
            else:
                losses["higher_aux"] = torch.tensor(0.0, device=logits.device)
        else:
            losses["focal"] = losses["supcon"] = losses["value_reg"] = losses["confusion_margin"] = None
            losses["semantic_anchor"] = losses["higher_aux"] = None

        return {
            "logits": logits,
            "losses": losses,
            "beta": beta.detach(),
            "proto_mix": gamma.detach(),
            "semantic_proto_mix": torch.sigmoid(self.semantic_proto_mix_logit).detach(),
            "labelwise_mix": eta.detach() if torch.is_tensor(eta) else torch.tensor(float(eta)),
            "dynamic_fusion_weights": None if dynamic_fusion_weights is None else dynamic_fusion_weights.detach(),
            "higher_logits": higher_logits.detach(),
            "pool_weights": pool_weights.detach(),
            "token_attn": token_attn.detach(),
            "gate_mean": gate.detach().mean(),
        }



# TRAINING HELPERS

def total_loss(output: dict, epoch_idx: int) -> torch.Tensor:
    warm = max(1, Config.CONTRASTIVE_WARMUP_EPOCHS)
    lam_con = Config.CONTRASTIVE_WEIGHT * min(1.0, float(epoch_idx + 1) / warm)
    lam_conf = Config.CONF_LOSS_WEIGHT if getattr(Config, "USE_CONFUSION_MARGIN", False) else 0.0
    losses = output["losses"]
    loss = (
        losses["focal"]
        + lam_con * losses["supcon"]
        + Config.VALUE_REG_WEIGHT * losses["value_reg"]
        + lam_conf * losses["confusion_margin"]
    )
    if getattr(Config, "USE_SEMANTIC_LABEL_PROTOTYPES", False):
        loss = loss + float(Config.SEMANTIC_ANCHOR_WEIGHT) * losses.get("semantic_anchor", loss.new_tensor(0.0))
    if getattr(Config, "USE_HIGHER_ORDER_HEAD", False):
        loss = loss + float(Config.HIGHER_AUX_WEIGHT) * losses.get("higher_aux", loss.new_tensor(0.0))
    return loss


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, epoch_idx: int = 10**6):
    model.eval()
    total_eval_loss = 0.0
    y_true, y_pred = [], []
    for batch in loader:
        ids = batch["input_ids"].to(Config.DEVICE)
        mask = batch["attention_mask"].to(Config.DEVICE)
        y = batch["labels"].to(Config.DEVICE)

        out = model(ids, mask, labels=y, epoch_idx=epoch_idx)
        loss = total_loss(out, epoch_idx)
        total_eval_loss += loss.item()

        pred = out["logits"].argmax(dim=1)
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(pred.detach().cpu().numpy())

    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics["loss"] = total_eval_loss / max(1, len(loader))
    return metrics, np.asarray(y_true), np.asarray(y_pred)


def train_model_with_val(train_loader: DataLoader, val_loader: DataLoader, seed: int, lr: float):
    set_seed(seed)
    model = ValueAwareGATBERT().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    fgm = FGM(model, emb_name=Config.FGM_EMB_NAME, epsilon=Config.FGM_EPS) if Config.USE_FGM else None

    best_macro, best_weighted, best_acc = -1.0, -1.0, -1.0
    best_epoch = 1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = {"train": [], "val": []}

    epoch_bar = trange(Config.EPOCHS, desc="VA-G1-FGM v3 Training", leave=True)
    for ep in epoch_bar:
        model.train()
        running_loss = 0.0
        all_y, all_p = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {ep + 1} Train", leave=False):
            ids = batch["input_ids"].to(Config.DEVICE)
            mask = batch["attention_mask"].to(Config.DEVICE)
            y = batch["labels"].to(Config.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            out = model(ids, mask, labels=y, epoch_idx=ep)
            clean_loss = total_loss(out, ep)
            clean_loss.backward()

            batch_loss_value = clean_loss.item()
            if fgm is not None:
                fgm.attack()
                adv_out = model(ids, mask, labels=y, epoch_idx=ep)
                adv_loss = total_loss(adv_out, ep)
                adv_loss.backward()
                fgm.restore()
                batch_loss_value = 0.5 * (clean_loss.item() + adv_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += batch_loss_value
            pred = out["logits"].argmax(dim=1)
            all_y.extend(y.detach().cpu().numpy())
            all_p.extend(pred.detach().cpu().numpy())

        train_metrics = compute_metrics(np.asarray(all_y), np.asarray(all_p))
        val_metrics, _, _ = evaluate_model(model, val_loader, epoch_idx=ep)

        history["train"].append({"loss": running_loss / max(1, len(train_loader)), **train_metrics})
        history["val"].append(val_metrics)

        epoch_bar.set_postfix({
            "val_macro_f1": f"{val_metrics['macro_f1']:.4f}",
            "val_acc": f"{val_metrics['accuracy']:.4f}",
        })

        if (
            (val_metrics["macro_f1"] > best_macro)
            or (val_metrics["macro_f1"] == best_macro and val_metrics["weighted_f1"] > best_weighted)
            or (val_metrics["macro_f1"] == best_macro and val_metrics["weighted_f1"] == best_weighted and val_metrics["accuracy"] > best_acc)
        ):
            best_macro = val_metrics["macro_f1"]
            best_weighted = val_metrics["weighted_f1"]
            best_acc = val_metrics["accuracy"]
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro),
        "best_weighted_f1": float(best_weighted),
        "best_accuracy": float(best_acc),
    }
    return model, summary, history


def train_fixed_epochs(train_loader: DataLoader, seed: int, lr: float, epochs: int):
    set_seed(seed)
    model = ValueAwareGATBERT().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    fgm = FGM(model, emb_name=Config.FGM_EMB_NAME, epsilon=Config.FGM_EPS) if Config.USE_FGM else None

    final_history = []
    epoch_bar = trange(epochs, desc="VA-G1-FGM v3 Final Retraining", leave=True)
    for ep in epoch_bar:
        model.train()
        running_loss = 0.0
        all_y, all_p = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {ep + 1} Train", leave=False):
            ids = batch["input_ids"].to(Config.DEVICE)
            mask = batch["attention_mask"].to(Config.DEVICE)
            y = batch["labels"].to(Config.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            out = model(ids, mask, labels=y, epoch_idx=ep)
            clean_loss = total_loss(out, ep)
            clean_loss.backward()

            batch_loss_value = clean_loss.item()
            if fgm is not None:
                fgm.attack()
                adv_out = model(ids, mask, labels=y, epoch_idx=ep)
                adv_loss = total_loss(adv_out, ep)
                adv_loss.backward()
                fgm.restore()
                batch_loss_value = 0.5 * (clean_loss.item() + adv_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += batch_loss_value
            pred = out["logits"].argmax(dim=1)
            all_y.extend(y.detach().cpu().numpy())
            all_p.extend(pred.detach().cpu().numpy())

        train_metrics = compute_metrics(np.asarray(all_y), np.asarray(all_p))
        train_loss = running_loss / max(1, len(train_loader))
        final_history.append({"epoch": ep + 1, "train_loss": train_loss, **train_metrics})
        epoch_bar.set_postfix({"train_loss": f"{train_loss:.4f}", "train_mf1": f"{train_metrics['macro_f1']:.4f}"})
    return model, final_history


def build_loader(texts: np.ndarray, labels: np.ndarray, tokenizer, train: bool, seed: int):
    aug_summary = {"mode": "none", "classes": {}}
    x_use, y_use = np.asarray(texts), np.asarray(labels)
    if train:
        x_use, y_use, aug_summary = augment_training_texts(x_use, y_use, seed=seed)

    dataset = TextDataset(x_use, y_use, tokenizer)
    if train and Config.USE_WEIGHTED_SAMPLER:
        sampler = make_weighted_sampler(y_use)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    return loader, aug_summary


def load_data():
    df = pd.read_csv(Config.DATA_PATH)
    df["category"] = df["category"].astype(str).map(normalize_label_name)
    csv_labels = set(df["category"].unique())
    cfg_labels = set(Config.LABEL_ORDER)
    assert csv_labels == cfg_labels, (
        "Dataset labels do not match Config.LABEL_ORDER after normalization.\n"
        f"CSV: {sorted(csv_labels)}\nCFG: {Config.LABEL_ORDER}"
    )
    label_map = {name: i for i, name in enumerate(Config.LABEL_ORDER)}
    df["label"] = df["category"].map(label_map)
    x = df["Base_Reviews"].fillna("none").values
    y = df["label"].values
    return x, y


# FIXED PSO CONFIGURATION — PSO SEARCH SKIPPED

def get_fixed_pso_config() -> Dict[str, float]:
    """Return the best PSO-selected configuration from the previous completed run."""
    return {k: float(v) for k, v in Config.BEST_PSO_CONFIG.items()}


def apply_fixed_pso_config(cfg: Optional[Dict[str, float]] = None) -> None:
    if cfg is None:
        cfg = get_fixed_pso_config()

    Config.WEIGHT_DECAY = float(cfg["weight_decay"])
    Config.DROPOUT = float(cfg["dropout"])
    Config.GAT_DROPOUT = float(cfg["gat_dropout"])
    Config.CONTRASTIVE_WEIGHT = float(cfg["contrastive_weight"])
    Config.VALUE_REG_WEIGHT = float(cfg["value_reg_weight"])
    Config.GRAPH_MIX_INIT = float(cfg["graph_mix_init"])
    Config.FGM_EPS = float(cfg["fgm_eps"])
    Config.LOGIT_SCALE_INIT = float(cfg["logit_scale_init"])


def fixed_config_search_summary() -> List[Dict[str, object]]:
    
    return [{
        "method": "fixed_previous_pso_config",
        "note": "PSO search skipped; configuration loaded from previous best PSO run.",
        "candidate_config": get_fixed_pso_config(),
    }]



# CROSS-VALIDATION

def run_cv(x_train_pool: np.ndarray, y_train_pool: np.ndarray, tokenizer, selected_lr: float):
    skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_metrics = []
    fold_selection = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train_pool, y_train_pool), start=1):
        print(f"\n[CV] Fold {fold}/{Config.CV_FOLDS}")
        x_fold_tr, y_fold_tr = x_train_pool[tr_idx], y_train_pool[tr_idx]
        x_fold_val, y_fold_val = x_train_pool[val_idx], y_train_pool[val_idx]

        train_loader, aug_summary = build_loader(x_fold_tr, y_fold_tr, tokenizer, train=True, seed=Config.SEED + 1000 + fold)
        val_loader, _ = build_loader(x_fold_val, y_fold_val, tokenizer, train=False, seed=Config.SEED)

        model, best_summary, history = train_model_with_val(train_loader, val_loader, Config.SEED + fold, selected_lr)
        val_metrics, _, _ = evaluate_model(model, val_loader, epoch_idx=10**6)
        fold_metrics.append({"fold": fold, **val_metrics})
        fold_selection.append({"fold": fold, "best_summary": best_summary, "augmentation": aug_summary, "history": history["val"]})

        print(f"Fold {fold} val macro-F1: {val_metrics['macro_f1']:.4f}")

    cv_summary = {
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean([x["accuracy"] for x in fold_metrics])),
        "std_accuracy": float(np.std([x["accuracy"] for x in fold_metrics])),
        "mean_macro_f1": float(np.mean([x["macro_f1"] for x in fold_metrics])),
        "std_macro_f1": float(np.std([x["macro_f1"] for x in fold_metrics])),
        "mean_weighted_f1": float(np.mean([x["weighted_f1"] for x in fold_metrics])),
        "std_weighted_f1": float(np.std([x["weighted_f1"] for x in fold_metrics])),
    }
    return cv_summary, fold_selection



# MAIN PIPELINE

def select_val_size() -> float:
    return Config.QUICK_MODEL_SELECTION_VAL_SIZE if Config.MODE == "quick" else Config.FULL_MODEL_SELECTION_VAL_SIZE


def run_pipeline():
    ensure_dir(Config.OUTPUT_DIR)
    set_seed(Config.SEED)

    x, y = load_data()
    tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_NAME)

    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x, y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.SEED,
    )

    #Fixed PSO-selected hyperparameters: skip expensive PSO search
    best_search_config = get_fixed_pso_config()
    apply_fixed_pso_config(best_search_config)
    best_lr = float(best_search_config["lr"])
    lr_results = fixed_config_search_summary()
    print("\n✅ Using fixed best PSO configuration:")
    print(json.dumps(best_search_config, indent=2))

    # Optional cross-validation
    cv_summary = None
    cv_details  = None
    if Config.MODE == "full":
        cv_summary, cv_details = run_cv(x_train_pool, y_train_pool, tokenizer, best_lr)

    # Model selection: train on sub-split to find best epoch
    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
        x_train_pool, y_train_pool,
        test_size=select_val_size(),
        stratify=y_train_pool,
        random_state=Config.SEED,
    )

    train_loader_sel, sel_aug = build_loader(x_train_final, y_train_final, tokenizer, train=True,  seed=Config.SEED + 101)
    val_loader_sel,   _       = build_loader(x_val_final,   y_val_final,   tokenizer, train=False, seed=Config.SEED)

    final_model_sel, selection_summary, selection_history = train_model_with_val(
        train_loader_sel, val_loader_sel, Config.SEED, best_lr
    )
    val_metrics, _, _ = evaluate_model(final_model_sel, val_loader_sel, epoch_idx=10**6)
    best_epoch   = selection_summary["best_epoch"]
    final_epochs = min(Config.EPOCHS, max(Config.FINAL_MIN_EPOCHS, best_epoch + Config.FINAL_EPOCH_BUFFER))
    print(
        f"\n🎯 Selected best_epoch={best_epoch} using validation macro-F1={selection_summary['best_macro_f1']:.4f}; "
        f"final retraining epochs={final_epochs}"
    )

    # Final model: retrain on the full train pool
    final_train_loader, final_aug = build_loader(x_train_pool, y_train_pool, tokenizer, train=True, seed=Config.SEED + 202)
    final_model, final_history    = train_fixed_epochs(final_train_loader, Config.SEED, best_lr, final_epochs)

    # Test evaluation
    test_loader, _ = build_loader(x_test, y_test, tokenizer, train=False, seed=Config.SEED)
    test_metrics, y_true, y_pred = evaluate_model(final_model, test_loader, epoch_idx=10**6)
    test_metrics_final = test_metrics

    if Config.MODE == "full" and cv_summary is not None:
        print(
            f"\n🟩 KGRV-former FULL | CV Macro-F1={cv_summary['mean_macro_f1']:.4f}±{cv_summary['std_macro_f1']:.4f} | "
            f"Test Macro-F1={test_metrics['macro_f1']:.4f} | Acc={test_metrics['accuracy']:.4f}"
        )
    else:
        print(
            f"\n🟩 KGRV-former| Macro-F1={test_metrics['macro_f1']:.4f} | "
            f"Weighted-F1={test_metrics['weighted_f1']:.4f} | Acc={test_metrics['accuracy']:.4f}"
        )
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=Config.DISPLAY_LABELS, digits=4))

    cm_final = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(
        cm_final,
        Config.DISPLAY_LABELS,
        Path(Config.OUTPUT_DIR) / "confusion_matrix_test.png",
        "KGRV-former Test Confusion Matrix (Schwartz Relation Graph Transformer)",
    )
    with open(Path(Config.OUTPUT_DIR) / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=Config.DISPLAY_LABELS, digits=4))

    save_learning_curves(selection_history, Path(Config.OUTPUT_DIR) / "learning_curves.png")

    # Persist everything
    summary = {
        "experiment_name": f"KGRV-former {Config.MODE} run + Schwartz relation-biased Graph Transformer",
        "selected_lr": float(best_lr),
        "selected_search_config": {k: float(v) for k, v in best_search_config.items()},
        "search_strategy": "fixed_previous_pso_config_no_search",
        "cv_mean_macro_f1": None if cv_summary is None else cv_summary["mean_macro_f1"],
        "cv_std_macro_f1": None if cv_summary is None else cv_summary["std_macro_f1"],
        "selection_summary": selection_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_final": test_metrics_final,
        "label_order": Config.LABEL_ORDER,
        "display_labels": Config.DISPLAY_LABELS,
        "lr_results": lr_results,
        "resampling": {
            "selection_augmentation": sel_aug,
            "final_augmentation": final_aug,
            "weighted_sampler": Config.USE_WEIGHTED_SAMPLER,
        },
        "changes": {
            "architecture": "BERT + semantic Schwartz prototypes + label-wise evidence attention + soft higher-order Schwartz auxiliary head + relation-biased label Graph Transformer",
            "pooling": "learnable last-4-layer masked-mean token mixing",
            "graph": "dense Schwartz relation-biased label Graph Transformer with learned attention bias for self, compatible, same higher-order, opposing, empirical-confusion, and neutral relations",
            "fusion": "gated residual relation-transformer node update + node logits mixed with direct semantic prototype-classifier logits",
            "value_regularizer": {
                "method": "target_similarity_matrix",
                "weight": Config.VALUE_REG_WEIGHT,
                "hard_targets": {f"{Config.LABEL_ORDER[a]}__{Config.LABEL_ORDER[b]}": v for (a, b), v in Config.STRUCTURE_HARD_TARGETS.items()},
            },
            "confusion_margin_loss": {
                "enabled": Config.USE_CONFUSION_MARGIN,
                "weight": Config.CONF_LOSS_WEIGHT,
                "margin": Config.CONF_MARGIN,
                "hard_negative_map": {
                    Config.LABEL_ORDER[k]: [Config.LABEL_ORDER[j] for j in v]
                    for k, v in Config.CONFUSION_HARD_NEGATIVES.items()
                },
            },
            "hard_triad_sampler": {
                "enabled": Config.USE_HARD_TRIAD_SAMPLER,
                "hard_triad_classes": [Config.LABEL_ORDER[i] for i in Config.HARD_TRIAD_CLASSES],
                "triad_boost": Config.HARD_TRIAD_BOOST,
                "conformity_boost": Config.HARD_CONFORMITY_BOOST,
            },
            "fgm": {
                "enabled": Config.USE_FGM,
                "epsilon": Config.FGM_EPS,
                "embedding_name": Config.FGM_EMB_NAME,
            },
            "final_retraining": {
                "best_epoch": int(best_epoch),
                "final_epochs": int(final_epochs),
                "epoch_buffer": int(Config.FINAL_EPOCH_BUFFER),
                "min_epochs": int(Config.FINAL_MIN_EPOCHS),
            },
            "true_final_retraining_on_all_X_tr": True,
            "kfold_used": Config.MODE == "full",
            "search_integration": "PSO search skipped; previously selected best PSO configuration applied directly",
            "step98_architecture_flags": {
                "semantic_label_prototypes": Config.USE_SEMANTIC_LABEL_PROTOTYPES,
                "labelwise_cross_attention": Config.USE_LABELWISE_CROSS_ATTENTION,
                "higher_order_aux_head": Config.USE_HIGHER_ORDER_HEAD,
                "semantic_proto_mix_init": Config.SEMANTIC_PROTO_MIX_INIT,
                "labelwise_mix_init": Config.LABELWISE_MIX_INIT,
                "higher_aux_weight": Config.HIGHER_AUX_WEIGHT,
                "higher_logit_mix": Config.HIGHER_LOGIT_MIX,
                "relation_graph_transformer": Config.USE_RELATION_GRAPH_TRANSFORMER,
                "relation_num_layers": Config.RELATION_NUM_LAYERS,
                "relation_num_heads": Config.RELATION_NUM_HEADS,
                "relation_dropout": Config.RELATION_DROPOUT,
                "sample_dynamic_fusion": getattr(Config, "USE_SAMPLE_DYNAMIC_FUSION", False),
                "dynamic_fusion_channels": getattr(Config, "DYNAMIC_FUSION_CHANNELS", 5),
            },
        },
    }

    if cv_summary is not None:
        json_dump(cv_summary, Path(Config.OUTPUT_DIR) / "cv_summary.json")
    if cv_details is not None:
        json_dump({"folds": cv_details}, Path(Config.OUTPUT_DIR) / "cv_details.json")
    json_dump(summary, Path(Config.OUTPUT_DIR) / "summary.json")
    json_dump(selection_history, Path(Config.OUTPUT_DIR) / "selection_history.json")
    json_dump({"epochs": final_history}, Path(Config.OUTPUT_DIR) / "final_history.json")


    print(f"\nArtifacts saved to: {Config.OUTPUT_DIR}")



# CLI

def apply_cli_overrides():
    parser = argparse.ArgumentParser(description="KGRV-former script with Schwartz relation-biased label Graph Transformer")
    parser.add_argument("--mode",           type=str,   choices=["quick", "full"], default=None)
    parser.add_argument("--data_path",      type=str,   default=None)
    parser.add_argument("--model_name",     type=str,   default=None)
    parser.add_argument("--output_dir",     type=str,   default=None)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--conf_loss_weight", type=float, default=None,
                        help="Weight for confusion-aware margin loss; set 0.0 to disable.")
    parser.add_argument("--conf_margin", type=float, default=None,
                        help="Margin used in the conformity/benevolence/hedonism hard-negative loss.")
    parser.add_argument("--conformity_boost", type=float, default=None,
                        help="Sampler multiplier for true conformity samples.")
    parser.add_argument("--triad_boost", type=float, default=None,
                        help="Sampler multiplier for true hedonism and benevolence samples.")
    parser.add_argument("--proto_mix_init", type=float, default=None,
                        help="Initial mix ratio for direct prototype-classifier branch.")
    parser.add_argument("--enable_two_step_edges", action="store_true",
                        help="Re-enable distance-2 graph edges; disabled by default.")
    parser.add_argument("--enable_conf_margin", action="store_true",
                        help="Enable weak confusion-margin loss. Disabled by default.")
    parser.add_argument("--enable_hard_triad_sampler", action="store_true",
                        help="Enable hard-triad sample boosting. Disabled by default.")
    parser.add_argument("--semantic_proto_mix_init", type=float, default=None,
                        help="Initial semantic/learned prototype mix ratio.")
    parser.add_argument("--semantic_anchor_weight", type=float, default=None,
                        help="Weight for semantic prototype anchoring loss.")
    parser.add_argument("--labelwise_mix_init", type=float, default=None,
                        help="Initial mix ratio for label-wise cross-attention branch.")
    parser.add_argument("--higher_aux_weight", type=float, default=None,
                        help="Auxiliary loss weight for higher-order Schwartz head.")
    parser.add_argument("--higher_logit_mix", type=float, default=None,
                        help="Logit prior strength from higher-order Schwartz head.")
    parser.add_argument("--relation_num_layers", type=int, default=None,
                        help="Number of relation-biased label Transformer layers.")
    parser.add_argument("--relation_num_heads", type=int, default=None,
                        help="Number of attention heads in the relation-biased label Transformer.")
    parser.add_argument("--relation_dropout", type=float, default=None,
                        help="Dropout inside the relation-biased label Transformer.")
    args = parser.parse_args()

    if args.mode is not None:
        Config.MODE = args.mode
    if args.data_path is not None:
        Config.DATA_PATH = args.data_path
    if args.model_name is not None:
        Config.MODEL_NAME = args.model_name
    if args.output_dir is not None:
        Config.OUTPUT_DIR = args.output_dir
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.seed is not None:
        Config.SEED = args.seed
    if args.conf_loss_weight is not None:
        Config.CONF_LOSS_WEIGHT = float(args.conf_loss_weight)
        Config.USE_CONFUSION_MARGIN = Config.CONF_LOSS_WEIGHT > 0.0
    if args.conf_margin is not None:
        Config.CONF_MARGIN = float(args.conf_margin)
    if args.conformity_boost is not None:
        Config.HARD_CONFORMITY_BOOST = float(args.conformity_boost)
    if args.triad_boost is not None:
        Config.HARD_TRIAD_BOOST = float(args.triad_boost)
    if args.proto_mix_init is not None:
        Config.PROTO_MIX_INIT = float(args.proto_mix_init)
    if args.enable_two_step_edges:
        Config.USE_TWO_STEP_GRAPH_EDGES = True
    if args.enable_conf_margin:
        Config.USE_CONFUSION_MARGIN = True
        if Config.CONF_LOSS_WEIGHT <= 0.0:
            Config.CONF_LOSS_WEIGHT = 0.03
    if args.enable_hard_triad_sampler:
        Config.USE_HARD_TRIAD_SAMPLER = True
        if Config.HARD_TRIAD_BOOST <= 1.0:
            Config.HARD_TRIAD_BOOST = 1.15
        if Config.HARD_CONFORMITY_BOOST <= 1.0:
            Config.HARD_CONFORMITY_BOOST = 1.3
    if args.semantic_proto_mix_init is not None:
        Config.SEMANTIC_PROTO_MIX_INIT = float(args.semantic_proto_mix_init)
    if args.semantic_anchor_weight is not None:
        Config.SEMANTIC_ANCHOR_WEIGHT = float(args.semantic_anchor_weight)
    if args.labelwise_mix_init is not None:
        Config.LABELWISE_MIX_INIT = float(args.labelwise_mix_init)
    if args.higher_aux_weight is not None:
        Config.HIGHER_AUX_WEIGHT = float(args.higher_aux_weight)
    if args.higher_logit_mix is not None:
        Config.HIGHER_LOGIT_MIX = float(args.higher_logit_mix)
    if args.relation_num_layers is not None:
        Config.RELATION_NUM_LAYERS = int(args.relation_num_layers)
    if args.relation_num_heads is not None:
        Config.RELATION_NUM_HEADS = int(args.relation_num_heads)
    if args.relation_dropout is not None:
        Config.RELATION_DROPOUT = float(args.relation_dropout)



if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    apply_cli_overrides()
    run_pipeline()