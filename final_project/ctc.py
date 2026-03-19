import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

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
    #print(pred_ids.count_nonzero()/pred_ids.shape[1])
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
# 5. beam search CTC decoding
# -----------------------------


NEG_INF = -1e30


def logsumexp2(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search_single(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0,
) -> List[int]:
    """
    log_probs: (T, V) for one utterance
    returns: best collapsed token sequence
    """
    if log_probs.ndim != 2:
        raise ValueError(f"Expected (T, V), got {tuple(log_probs.shape)}")

    T, V = log_probs.shape

    # beams[prefix] = (p_blank, p_nonblank)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        (): (0.0, NEG_INF)
    }

    for t in range(T):
        next_beams = defaultdict(lambda: (NEG_INF, NEG_INF))
        frame = log_probs[t]  # (V,)

        # optional frame pruning
        topk = min(beam_width, V)
        _, top_ids = torch.topk(frame, k=topk)

        for prefix, (p_b, p_nb) in beams.items():
            # 1) emit blank
            p_blank = float(frame[blank_id].item())
            nb_pb, nb_pnb = next_beams[prefix]
            nb_pb = logsumexp2(nb_pb, p_b + p_blank)
            nb_pb = logsumexp2(nb_pb, p_nb + p_blank)
            next_beams[prefix] = (nb_pb, nb_pnb)

            # 2) emit non-blank
            last = prefix[-1] if len(prefix) > 0 else None

            for s in top_ids.tolist():
                if s == blank_id:
                    continue

                p_s = float(frame[s].item())

                if s == last:
                    # (a) continue repeated symbol without changing collapsed prefix
                    nb_pb, nb_pnb = next_beams[prefix]
                    nb_pnb = logsumexp2(nb_pnb, p_nb + p_s)
                    next_beams[prefix] = (nb_pb, nb_pnb)

                    # (b) append same symbol only if previous path ended with blank
                    new_prefix = prefix + (s,)
                    nb_pb2, nb_pnb2 = next_beams[new_prefix]
                    nb_pnb2 = logsumexp2(nb_pnb2, p_b + p_s)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)
                else:
                    new_prefix = prefix + (s,)
                    nb_pb2, nb_pnb2 = next_beams[new_prefix]
                    nb_pnb2 = logsumexp2(nb_pnb2, p_b + p_s)
                    nb_pnb2 = logsumexp2(nb_pnb2, p_nb + p_s)
                    next_beams[new_prefix] = (nb_pb2, nb_pnb2)

        # prune
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda x: logsumexp2(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(
        beams.items(),
        key=lambda x: logsumexp2(x[1][0], x[1][1]),
    )[0]

    return list(best_prefix)

def ctc_beam_decode(
    log_probs: torch.Tensor,
    input_lens: torch.Tensor,
    blank_id: int,
    beam_width: int = 10,
) -> List[List[int]]:
    """
    Args:
        log_probs: (T, B, V)
        input_lens: (B,)
    Returns:
        List of decoded token-id sequences
    """
    if log_probs.ndim != 3:
        raise ValueError(f"Expected log_probs shape (T, B, V), got {tuple(log_probs.shape)}")

    T, B, V = log_probs.shape
    decoded = []

    for b in range(B):
        Tb = int(input_lens[b].item())
        lp = log_probs[:Tb, b, :]  # (Tb, V)
        seq = ctc_prefix_beam_search_single(
            log_probs=lp,
            beam_width=beam_width,
            blank_id=blank_id,
        )
        decoded.append(seq)

    return decoded
# -----------------------------
# 6. Error rate utilities
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

@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    features: torch.Tensor,
    feature_lens: torch.Tensor,
    tokenizer,
    device: torch.device,
    return_log_probs: bool = False,
    return_confidence: bool = False,
    decode_method: str = "beam",   # "beam" or "greedy"
    beam_width: int = 10,
):
    model.eval()

    features = features.to(device)
    logits = model(features, feature_lens)          # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)       # (B, T, V)
    log_probs_t = log_probs.transpose(0, 1)         # (T, B, V)

    if decode_method == "greedy":
        pred_token_ids = ctc_greedy_decode(
            log_probs=log_probs_t.cpu(),
            input_lens=feature_lens.cpu(),
            blank_id=tokenizer.blank_id,
        )
    elif decode_method == "beam":
        pred_token_ids = ctc_beam_decode(
            log_probs=log_probs_t.cpu(),
            input_lens=feature_lens.cpu(),
            blank_id=tokenizer.blank_id,
            beam_width=beam_width,
        )
    else:
        raise ValueError(f"Unknown decode_method: {decode_method}")

    pred_texts = [tokenizer.decode(ids) for ids in pred_token_ids]

    out = {
        "texts": pred_texts,
        "token_ids": pred_token_ids,
    }

    if return_confidence:
        probs = log_probs.exp()
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
    decode_method: str = "beam",
    beam_width: int = 10,
):
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
        decode_method=decode_method,
        beam_width=beam_width,
    )

    out = {
        "text": batch_out["texts"][0],
        "token_ids": batch_out["token_ids"][0],
    }

    if return_confidence:
        out["confidence"] = batch_out["confidence"][0]

    if return_log_probs:
        out["log_probs"] = batch_out["log_probs"][0]

    return out


