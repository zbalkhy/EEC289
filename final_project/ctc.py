import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PHONEMES = [
'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY',
'OW','OY','UH','UW',
'P','B','T','D','K','G',
'CH','JH',
'F','V','TH','DH','S','Z','SH','ZH','HH',
'M','N','NG',
'L','R','W','Y', " "
]

# -----------------------------
# 2. Tokenizer
# -----------------------------
class CharTokenizer:
    """
    Very simple character tokenizer for CTC.
    Index 0 is reserved for blank.
    """
    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz '", phonemes=False):
        self.blank_id = 0
        self.phonemes = phonemes
        self.alphabet = alphabet
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.id_to_char = {i + 1: ch for i, ch in enumerate(alphabet)}
        self.vocab_size = len(alphabet) + 1  # + blank

    def encode(self, text: str) -> List[int]:
        if not self.phonemes:
            text = text.lower()
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.phonemes:
            return [self.id_to_char[i] for i in ids if i in self.id_to_char]
        else:
            return "".join(self.id_to_char[i] for i in ids if i in self.id_to_char)


# -----------------------------
# 3. Collate function
# -----------------------------
def ctc_collate_fn(batch: List[Dict], tokenizer: CharTokenizer) -> Dict[str, torch.Tensor]:
    """
    Expected per-sample format:
    {
        "features": FloatTensor (T, D),   # SMT embedding sequence
        "text": str
    }
    """
    batch_size = len(batch)
    feat_lens = [item["features"].shape[0] for item in batch]
    d_in = batch[0]["features"].shape[1]
    max_t = max(feat_lens)

    feats = torch.zeros(batch_size, max_t, d_in, dtype=torch.float32)
    feat_lens_tensor = torch.tensor(feat_lens, dtype=torch.long)

    all_targets = []
    target_lens = []

    for i, item in enumerate(batch):
        x = item["features"]
        t = x.shape[0]
        feats[i, :t] = x

        target_ids = tokenizer.encode(item["text"])
        all_targets.extend(target_ids)
        target_lens.append(len(target_ids))

    targets = torch.tensor(all_targets, dtype=torch.long)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return {
        "features": feats,               # (B, T, D)
        "feature_lens": feat_lens_tensor,
        "targets": targets,              # (sum target lengths,)
        "target_lens": target_lens,      # (B,)
        "texts": [item["text"] for item in batch],
    }


# -----------------------------
# 4. Greedy CTC decoding
# -----------------------------
def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lens: torch.Tensor,
    blank_id: int,
) -> List[List[int]]:
    """
    log_probs: (T, B, V)
    input_lens: (B,)
    returns: list of token-id sequences after CTC collapse
    """
    pred_ids = log_probs.argmax(dim=-1)  # (T, B)
    pred_ids = pred_ids.transpose(0, 1)  # (B, T)

    decoded = []
    for b in range(pred_ids.shape[0]):
        seq = pred_ids[b, :input_lens[b]].tolist()
        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        decoded.append(collapsed)
    return decoded


# -----------------------------
# 5. Error rate utilities
# -----------------------------
def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """
    Token-level Levenshtein distance.
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


def char_error_rate(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


from typing import List, Dict, Union, Optional
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    features: torch.Tensor,
    feature_lens: torch.Tensor,
    tokenizer,
    device: torch.device,
    return_log_probs: bool = False,
    return_confidence: bool = False,
) -> Dict[str, Union[List[str], List[List[int]], torch.Tensor, List[float]]]:
    """
    Greedy CTC decoding for a batch.

    Args:
        model:
            Trained SMTCTCLinear model.
        features:
            Tensor of shape (B, T, D)
        feature_lens:
            Tensor of shape (B,) with valid frame counts
        tokenizer:
            Must provide blank_id and decode(ids)
        device:
            cpu or cuda
        return_log_probs:
            If True, include frame-level log_probs (B, T, V) on CPU
        return_confidence:
            If True, include a simple confidence proxy per utterance

    Returns:
        dict with:
            "texts": List[str]
            "token_ids": List[List[int]]
            optionally "log_probs": Tensor (B, T, V)
            optionally "confidence": List[float]
    """
    model.eval()

    features = features.to(device)
    logits = model(features, feature_lens)                    # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
    log_probs_t = log_probs.transpose(0, 1)     # (T, B, V)

    pred_token_ids = ctc_greedy_decode(
        log_probs=log_probs_t.cpu(),
        input_lens=feature_lens.cpu(),
        blank_id=tokenizer.blank_id,
    )
    pred_texts = [tokenizer.decode(ids) for ids in pred_token_ids]

    out = {
        "texts": pred_texts,
        "token_ids": pred_token_ids,
    }

    if return_confidence:
        # Simple confidence proxy:
        # average max posterior over valid frames
        probs = log_probs.exp()  # (B, T, V)
        confs = []
        for b in range(probs.shape[0]):
            T = int(feature_lens[b].item())
            if T == 0:
                confs.append(0.0)
                continue
            max_post = probs[b, :T].max(dim=-1).values
            confs.append(float(max_post.mean().item()))
        out["confidence"] = confs

    if return_log_probs:
        out["log_probs"] = log_probs.cpu()

    return out


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    features: torch.Tensor,
    tokenizer,
    device: torch.device,
    return_log_probs: bool = False,
    return_confidence: bool = False,
) -> Dict[str, Union[str, List[int], torch.Tensor, float]]:
    """
    Greedy CTC decoding for a single utterance.

    Args:
        features:
            Tensor of shape (T, D)

    Returns:
        dict with:
            "text": str
            "token_ids": List[int]
            optionally "log_probs": Tensor (T, V)
            optionally "confidence": float
    """
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (T, D), got {tuple(features.shape)}")

    batch_features = features.unsqueeze(0)  # (1, T, D)
    feature_lens = torch.tensor([features.shape[0]], dtype=torch.long)

    batch_out = predict_batch(
        model=model,
        features=batch_features,
        feature_lens=feature_lens,
        tokenizer=tokenizer,
        device=device,
        return_log_probs=return_log_probs,
        return_confidence=return_confidence,
    )

    out = {
        "text": batch_out["texts"][0],
        "token_ids": batch_out["token_ids"][0],
    }

    if return_confidence:
        out["confidence"] = batch_out["confidence"][0]

    if return_log_probs:
        out["log_probs"] = batch_out["log_probs"][0]  # (T, V)

    return out