"""
IDBR: Identity-Disentangled Battery Representation
====================================================

Package implementing the IDBR framework described in:
"Identity-Preserving Deep Learning for Lithium-Ion Battery Prognostics
and Telemetry Security" (Malhotra, Patidar, Mishra, Sarvanakumar).

Modules
-------
preprocessing   : NASA PCoE parsing, cleaning, resampling, normalization
segmentation    : fixed-length windowing of cycle telemetry
augmentation    : sensor noise, temporal warping, random cropping
data_loader     : cell-aware PyTorch Dataset/DataLoader utilities
model           : ConvTransformerEncoder, GRL, IDBRModel, BaselineModel
losses          : InfoNCE + degradation + adversarial disentanglement loss
train           : full training loop (AdamW, cosine LR, GRL ramp)
authentication  : fingerprint DB, EER / FAR / FRR / Top-1 retrieval
tamper          : replay / interpolation / value-injection attack engine
evaluate        : evaluation suite (plots, CSV tables, ablation grid)
"""

__version__ = "1.0.0"

__all__ = [
    "preprocessing",
    "segmentation",
    "augmentation",
    "data_loader",
    "model",
    "losses",
    "train",
    "authentication",
    "tamper",
    "evaluate",
]
