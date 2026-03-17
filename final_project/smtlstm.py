import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ctc import * 

class SMTCTCBiLSTM(nn.Module):
    def __init__(
        self,
        d_in: int,
        vocab_size: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        """
        1-layer BiLSTM CTC model.

        Args:
            d_in: input SMT feature dimension
            vocab_size: number of output tokens including blank
            hidden_size: LSTM hidden size per direction
            dropout: input/output dropout
            use_layernorm: whether to apply LayerNorm to input features
        """
        super().__init__()

        self.ln = nn.LayerNorm(d_in) if use_layernorm else nn.Identity()
        self.in_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, z: torch.Tensor, z_lens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, T, D)
            z_lens: (B,) optional valid lengths

        Returns:
            logits: (B, T, V)
        """
        z = self.ln(z)
        z = self.in_drop(z)

        if z_lens is not None:
            # pack padded sequence so the LSTM ignores padded frames
            packed = nn.utils.rnn.pack_padded_sequence(
                z,
                lengths=z_lens.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=z.shape[1],
            )
        else:
            lstm_out, _ = self.lstm(z)

        lstm_out = self.out_drop(lstm_out)
        logits = self.proj(lstm_out)  # (B, T, V)
        return logits
    
# -----------------------------
# 6. Train / eval step
# -----------------------------
@dataclass
class EpochStats:
    loss: float
    cer: float
    wer: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    tokenizer: CharTokenizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    grad_clip_norm: float = 5.0,
) -> EpochStats:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    total_cer = 0.0
    total_wer = 0.0
    total_examples = 0

    for batch in loader:
        feats = batch["features"].to(device)                 # (B, T, D)
        feat_lens = batch["feature_lens"]                    # keep on CPU for CTCLoss compatibility
        targets = batch["targets"]                           # keep on CPU for CTCLoss compatibility
        target_lens = batch["target_lens"]                   # keep on CPU for CTCLoss compatibility
        texts = batch["texts"]

        logits = model(feats, feat_lens)                               # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)            # (B, T, V)
        log_probs_t = log_probs.transpose(0, 1)              # (T, B, V)

        loss = ctc_loss_fn(
            log_probs_t,
            targets,
            feat_lens,
            target_lens,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            pred_token_ids = ctc_greedy_decode(
                log_probs_t.detach().cpu(),
                feat_lens.cpu(),
                blank_id=tokenizer.blank_id,
            )
            pred_texts = [tokenizer.decode(ids) for ids in pred_token_ids]

            batch_cer = 0.0
            batch_wer = 0.0
            for ref, hyp in zip(texts, pred_texts):
                ref_norm = ref #ref.lower()
                batch_cer += char_error_rate(ref_norm, hyp)
                #batch_wer += word_error_rate(ref_norm, hyp)

            bsz = len(texts)
            total_cer += batch_cer
            total_wer += batch_wer
            total_examples += bsz

        total_loss += loss.item()
        total_batches += 1

    return EpochStats(
        loss=total_loss / max(total_batches, 1),
        cer=total_cer / max(total_examples, 1),
        wer=total_wer / max(total_examples, 1),
    )


# -----------------------------
# 7. Full training loop
# -----------------------------
def train_ctc_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 5.0,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    ctc_loss_fn = nn.CTCLoss(
        blank=tokenizer.blank_id,
        reduction="mean",
        zero_infinity=True,
    )

    best_val_cer = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            ctc_loss_fn=ctc_loss_fn,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                ctc_loss_fn=ctc_loss_fn,
                tokenizer=tokenizer,
                device=device,
                optimizer=None,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats.loss:.4f} "
            f"train_CER={train_stats.cer:.4f} "
            f"train_WER={train_stats.wer:.4f} || "
            f"val_loss={val_stats.loss:.4f} "
            f"val_CER={val_stats.cer:.4f} "
            f"val_WER={val_stats.wer:.4f}"
        )

        if val_stats.cer < best_val_cer:
            best_val_cer = val_stats.cer
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_cer": val_stats.cer,
                "val_wer": val_stats.wer,
            }

    return best_state

def load_trained_bilstm_model(
    checkpoint_path: str,
    d_in: int,
    vocab_size: int,
    device: torch.device,
    hidden_size: int = 256,
    dropout: float = 0.1,
    use_layernorm: bool = True,
):
    model = SMTCTCBiLSTM(
        d_in=d_in,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        dropout=dropout,
        use_layernorm=use_layernorm,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model