from pathlib import Path
import torch
import numpy as np
from SMT import *
import torchaudio
from kmeans import *
from ctc import *
from smtlstm import *
from smt_accumulator import *
import os



def extract_and_save_features(ds, dir, n_mels, patch_ms, window_ms=10, hop_ms=5):
    audio_features = extract_features_from_dataset(ds, 
                                                len(ds), 
                                                n_mels,
                                                window_ms=window_ms, hop_ms=hop_ms)

    # this is kinda stupid, just do this in the previous function
    log_mels = [clip["log_mel_db"] for _, clip in audio_features.items()]
    text = [clip['text'] for _, clip in audio_features.items()]

    hop_length = [clip["hop_length"] for _, clip in audio_features.items()][0]
    sr = [clip["sr"] for _, clip in audio_features.items()][0]

    # patch and preprocess
    patches, utterance_bounds = patch_multiple_utterances(log_mels, sr, hop_length, patch_ms, patch_frame_hop=1)
    norm_patches, mean, std, zcaMatrix = preprocess_patches(patches)

    # save everything
    np.save(os.path.join(dir,"norm_patches.npy"), norm_patches)
    np.save(os.path.join(dir,"text.npy"), text)
    np.save(os.path.join(dir,"utterance_bounds.npy"), utterance_bounds)
    np.save(os.path.join(dir,"mean.npy"), mean)
    np.save(os.path.join(dir,"std.npy"), std)
    np.save(os.path.join(dir,"zcaMatrix.npy"), zcaMatrix)
    
    return norm_patches, utterance_bounds

def kmeans_and_save(norm_patches, dir):

    # kmeans for sparse coding
    labels, centroids, inertia, counts = minibatch_kmeans_gpu_vectorized(
        norm_patches,
        n_clusters=3000,
        batch_size=8000,
        n_iters=1000,
        n_init=3,
        distance="euclidean",
        init="random",
        reassignment_ratio=0.01,
        reassignment_freq=25,
        final_reassign=True,
        final_batch_size=8000,
        verbose=False,
    )
    np.save(os.path.join(dir,'clusters.npy'), centroids.cpu().numpy())
    return centroids

def sparse_coding(centroids, norm_patches, diff_order, utterance_bounds, device, dir):
    accumulator = DenseKSparseSMTAccumulator(centroids,
                                         K=16,
                                         batch_size=100,
                                         enforce_positive_coefficients=True,
                                         normalize_dictionary=True,
                                         FISTA_Iters=1000,
                                         diff_order=diff_order,
                                         device=device,
                                         dtype=torch.float32,
                                         center_codes=False)
    accumulator.batch_process_patches(torch.tensor(norm_patches), utterance_bounds)
    
    smt_stats = accumulator.finalize()
    sparse_codes = torch.cat([batch.to_dense() for batch in accumulator.sparse_codes_per_batch])

    torch.save(smt_stats, os.path.join(dir,"smt_stats.pt"))
    torch.save(sparse_codes, os.path.join(dir,"sparse_codes.pt"))

if __name__ == "__main__":

    save_dir = "/mnt/data/SMT/"

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set dataset
    librispeech_ds = torchaudio.datasets.LIBRISPEECH(
    root=str("/mnt/data/SMT"), url="train-clean-100", download=True,
    )

    total_samples = 2000
    sample = torch.randperm(len(librispeech_ds))[:total_samples]
    sample_ds = [librispeech_ds[idx] for idx in sample]

    diff_orders = ["first", "second"]
    patch_mss = [50, 80, 100]

    for patch_ms in patch_mss:
        dir = os.path.join(save_dir, "_" + str(patch_ms))
        try:
            os.mkdir(os.path.join(dir))
            print(f"Directory '{dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir}' already exists.")
        
        norm_patches, utterance_bounds = extract_and_save_features(sample_ds, dir, 20, patch_ms)
        torch.cuda.empty_cache()
        
        centroids = kmeans_and_save(norm_patches, dir)
        torch.cuda.empty_cache()

        for diff_order in diff_orders:
            diff_dir = os.path.join(dir, diff_order)
            try:
                os.mkdir(os.path.join(diff_dir))
                print(f"Directory '{diff_dir}' created successfully.")
            except FileExistsError:
                print(f"Directory '{diff_dir}' already exists.")
            sparse_coding(centroids, norm_patches, diff_order, utterance_bounds, device, diff_dir)
            torch.cuda.empty_cache()