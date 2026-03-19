from pathlib import Path
import zipfile
import librosa
import torch
import numpy as np
from SMT import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm
import json
import torchaudio
from kmeans import *
from ctc import *
from smtlstm import *
from smt_accumulator import *
import os
import re
from g2p_en import G2p

def make_dataloader(dataset, tokenizer, batch_size=16, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: ctc_collate_fn(batch, tokenizer),
    )

class NormBetaDataset(Dataset):
    """Simple utterance-level dataset built from norm_beta and utterance bounds."""
    def __init__(self, norm_beta: np.ndarray, utterance_bounds, texts=None, phonemes: bool = False):
        self.norm_beta = norm_beta
        self.utterance_bounds = list(utterance_bounds)
        self.g2p = G2p()
        self.phonemes = phonemes
        if texts is None:
            self.texts = [""] * len(self.utterance_bounds)
        else:
            if len(texts) != len(self.utterance_bounds):
                raise ValueError("texts must match utterance_bounds length")
            self.texts = list(texts)

    def __len__(self):
        return len(self.utterance_bounds)

    def __getitem__(self, idx):
        start, end = self.utterance_bounds[idx]
        x = torch.as_tensor(self.norm_beta[start:end, :], dtype=torch.float32)
        targets = self._text_to_phonemes(self.texts[idx]) if self.phonemes else self.texts[idx]
        return {
            "features": x,
            "text": targets,
        }
    
    def _text_to_phonemes(self, text, remove_stress=True):
        phonemes = self.g2p(text)

        # remove spaces/punctuation tokens
        phonemes = [p for p in phonemes if p.strip()]

        if remove_stress:
            phonemes = [p.rstrip("012") for p in phonemes]

        return phonemes
    
def solve_SMT_matrix(smt_stats: SMTStats):
    V = smt_stats.V
    M = smt_stats.M
    # ------------------------------------------------------------
    # Solve generalized eigenproblem M u = λ V u
    # via Cholesky reduction
    # ------------------------------------------------------------
    L = torch.linalg.cholesky(V)

    # C = L^{-1} M L^{-T}
    Linv_M = torch.linalg.solve_triangular(L, M, upper=False, left=True)
    C = torch.linalg.solve_triangular(
        L, Linv_M.transpose(-1, -2), upper=False, left=True
    ).transpose(-1, -2)

    # Symmetrize for numerical stability
    C = 0.5 * (C + C.T)

    eigvals, y = torch.linalg.eigh(C)

    # Recover generalized eigenvectors
    eigvecs = torch.linalg.solve_triangular(L.T, y, upper=True, left=True)
    return eigvals, eigvecs

def train_model(dataset, 
                train_bounds,
                val_bounds,
                train_text,
                val_text, 
                dir):
    # train log mel model on norm patches
    train_dataset = NormBetaDataset(dataset, train_bounds, train_text, phonemes=False)
    val_dataset = NormBetaDataset(dataset, val_bounds, val_text, phonemes=False)

    tokenizer = CharTokenizer()
    train_loader = make_dataloader(train_dataset, tokenizer, batch_size=100, shuffle=True)
    val_loader = make_dataloader(val_dataset, tokenizer, batch_size=100, shuffle=False)

    # Infer input dimension from one sample
    sample = train_dataset[0]["features"]
    d_in = sample.shape[1]

    model = SMTCTCBiLSTM(
        d_in=d_in,
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        dropout=0.1,
        use_layernorm=True,
    )

    best_state, train_stats = train_ctc_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        num_epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=5.0,
        decode_method="greedy"
    )
    return best_state, train_stats


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_dir = "/mnt/data/SMT"
    patch_sizes = ['_50', '_80', '_100']
    diff_orders = ['first', 'second']
    data_sizes = [100, 500, 1000, 2000]

    for patch_size in patch_sizes:
        patch_size_dir = os.path.join(main_dir, patch_size)

        norm_patches = np.load(os.path.join(patch_size_dir,"norm_patches.npy"))
        utterance_bounds = np.load(os.path.join(patch_size_dir,"utterance_bounds.npy"))
        text = np.load(os.path.join(patch_size_dir,"text.npy"))
        mean = np.load(os.path.join(patch_size_dir,"mean.npy"))
        std = np.load(os.path.join(patch_size_dir,"std.npy"))
        zcaMatrix = np.load(os.path.join(patch_size_dir,"zcaMatrix.npy"))

        for size in data_sizes:
            data_length = size
            train_split =int(data_length*0.85)
            val_split = int(data_length*0.15)
            
            train_bounds = utterance_bounds[:train_split]
            train_text = text[:train_split]
            
            # ensure we are always using same validation set for better comparisons
            val_bounds = utterance_bounds[-train_split:]
            val_text = text[-train_split:]
            

            # train log mel model on norm patches
            best_state, train_stats = train_model(norm_patches,
                                                  train_bounds,
                                                  val_bounds,
                                                  train_text,
                                                  val_text,
                                                  patch_size_dir)
            if best_state is not None:
                torch.save(best_state, os.path.join(patch_size_dir, f"mel_{patch_size}_{str(size)}_ctc.pt"))
                print(f"Saved best mel for patchsize checkpoint with val CER={best_state['val_cer']:.4f}")
    
            # Define the filename
            filename = f'{patch_size}_{str(size)}_mel_training_stats.json'

            # Open the file in write mode ('w') and use json.dump()
            with open(filename, 'w') as json_file:
                json.dump(train_stats, json_file, indent=4) # The indent parameter adds formatting for human-readabilit
            torch.cuda.empty_cache()
            # train first and second order difference on smt model
            for difference in diff_orders:
                diff_dir = os.path.join(patch_size_dir, difference)
                smt_stats = torch.load(os.path.join(diff_dir,"smt_stats.pt"), weights_only=False)
                sparse_codes = torch.load(os.path.join(diff_dir,"sparse_codes.pt"))
                
                eigvals, eigvecs = solve_SMT_matrix(smt_stats)
                d = norm_patches.shape[1] # keep smt representation size matched with mel
                P = eigvecs[:, 1:d+1].T

                beta = P.cpu()@sparse_codes.T
                norm_beta, _, _, _ = preprocess_patches(beta.T.cpu().numpy()) # this switches beta to be NxD like norm_patches
                norm_beta = norm_beta

                best_state, train_stats = train_model(norm_beta,
                                                  train_bounds,
                                                  val_bounds,
                                                  train_text,
                                                  val_text,
                                                  patch_size_dir)
                if best_state is not None:
                    torch.save(best_state, os.path.join(patch_size_dir, f"smt_{patch_size}_{str(size)}_{difference}_ctc.pt"))
                    print(f"Saved best mel for patchsize checkpoint with val CER={best_state['val_cer']:.4f}")
        
                # Define the filename
                filename = f'{patch_size}_{str(size)}_{difference}_smt_training_stats.json'

                # Open the file in write mode ('w') and use json.dump()
                with open(filename, 'w') as json_file:
                    json.dump(train_stats, json_file, indent=4) # The indent parameter adds formatting for human-readabilit

                torch.cuda.empty_cache()
                del norm_beta
