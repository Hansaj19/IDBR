"""
src/authentication.py
=======================
Implements the authentication protocol of Section IV-A / Algorithm 2:

- ``Enroller``: computes a cell's fingerprint as the mean (L2-renormalized)
  identity embedding across its enrollment windows (Eq. "enrollment"
  step of Algorithm 2).
- ``FingerprintDatabase``: an append-only store keyed by cell id, matching
  the deployment consideration in Section V-D ("enrollment records cannot
  be silently overwritten").
- ``compute_eer``: solves for the Equal Error Rate threshold where
  FAR(tau) = FRR(tau) (Eq. 9), via ROC-curve interpolation.
- ``compute_far_frr``: FAR/FRR at a specific threshold.
- ``compute_roc_auc``: ROC-AUC for genuine vs impostor score distributions.
- ``top1_retrieval_accuracy``: nearest-fingerprint retrieval accuracy
  against the database (Algorithm 2, "Retrieval mode").
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

class Enroller:
    """Computes a cell fingerprint as the L2-normalized mean of a set of
    identity embeddings (Algorithm 2, enrollment step)."""

    @staticmethod
    def enroll(embeddings: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        embeddings : np.ndarray
            Shape (K, D) of L2-normalized identity embeddings for K
            enrollment windows belonging to one physical cell.

        Returns
        -------
        np.ndarray of shape (D,)
            The L2-normalized mean fingerprint.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("enroll() requires a non-empty (K, D) array.")
        mean_vec = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-12:
            return mean_vec
        return mean_vec / norm


# ---------------------------------------------------------------------------
# Fingerprint database (append-only)
# ---------------------------------------------------------------------------

class FingerprintDatabase:
    """Append-only fingerprint store keyed by cell serial number/id. Once
    a cell id is enrolled it cannot be silently overwritten -- callers
    must explicitly call :meth:`update` to replace an existing record,
    matching the deployment requirement in Section V-D of the paper.
    """

    def __init__(self):
        self._db: Dict[str, np.ndarray] = {}
        self._enrollment_log: List[dict] = []

    def add(self, cell_id: str, fingerprint: np.ndarray) -> None:
        if cell_id in self._db:
            raise ValueError(
                f"Cell '{cell_id}' is already enrolled. The fingerprint "
                f"database is append-only; use update() to explicitly "
                f"replace an existing enrollment record."
            )
        self._db[cell_id] = np.asarray(fingerprint, dtype=np.float32)
        self._enrollment_log.append({"cell_id": cell_id, "action": "add"})

    def update(self, cell_id: str, fingerprint: np.ndarray) -> None:
        self._db[cell_id] = np.asarray(fingerprint, dtype=np.float32)
        self._enrollment_log.append({"cell_id": cell_id, "action": "update"})

    def get(self, cell_id: str) -> np.ndarray:
        return self._db[cell_id]

    def cells(self) -> List[str]:
        return list(self._db.keys())

    def __contains__(self, cell_id: str) -> bool:
        return cell_id in self._db

    def __len__(self) -> int:
        return len(self._db)

    def matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Returns (D, N) stacked fingerprints and the corresponding cell
        id order, for vectorized retrieval search."""
        cell_ids = self.cells()
        if not cell_ids:
            return np.zeros((0, 0), dtype=np.float32), []
        mat = np.stack([self._db[c] for c in cell_ids], axis=0)  # (N, D)
        return mat, cell_ids

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serializable = {cid: vec.tolist() for cid, vec in self._db.items()}
        with open(path, "w") as f:
            json.dump({"fingerprints": serializable, "log": self._enrollment_log}, f)

    @staticmethod
    def load(path: str) -> "FingerprintDatabase":
        with open(path, "r") as f:
            payload = json.load(f)
        db = FingerprintDatabase()
        for cid, vec in payload["fingerprints"].items():
            db._db[cid] = np.array(vec, dtype=np.float32)
        db._enrollment_log = payload.get("log", [])
        return db


# ---------------------------------------------------------------------------
# Similarity / matching
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def match(query: np.ndarray, enrolled: np.ndarray, threshold: float) -> bool:
    """Implements Eq. (8): match iff cosine similarity >= threshold."""
    return cosine_similarity(query, enrolled) >= threshold


# ---------------------------------------------------------------------------
# FAR / FRR / EER
# ---------------------------------------------------------------------------

def compute_far_frr(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """FAR = fraction of impostor scores >= threshold (falsely accepted).
    FRR = fraction of genuine scores < threshold (falsely rejected)."""
    far = float(np.mean(impostor_scores >= threshold)) if len(impostor_scores) else float("nan")
    frr = float(np.mean(genuine_scores < threshold)) if len(genuine_scores) else float("nan")
    return far, frr


def compute_eer(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray
) -> Tuple[float, float]:
    """Solves for the Equal Error Rate operating point (Eq. 9) via the
    ROC curve of the genuine-vs-impostor binary detection problem.

    Returns
    -------
    (eer, threshold) : Tuple[float, float]
        The equal error rate (in [0, 1]) and the similarity threshold
        at which FAR == FRR (found by interpolation of the ROC curve).
    """
    labels = np.concatenate(
        [np.ones_like(genuine_scores), np.zeros_like(impostor_scores)]
    )
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr  # FRR
    # FAR == fpr. Find the point where fpr == fnr via interpolation.
    diff = fpr - fnr
    idx = np.nanargmin(np.abs(diff))

    if idx == 0 or idx == len(diff) - 1:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_threshold = float(thresholds[idx])
        return eer, eer_threshold

    # linear interpolation between idx-1 and idx (or idx and idx+1) for a
    # more precise EER estimate.
    if diff[idx] == 0:
        eer = float(fpr[idx])
        eer_threshold = float(thresholds[idx])
    else:
        # find the neighboring index with opposite sign of diff
        neighbor = idx + 1 if idx + 1 < len(diff) and np.sign(diff[idx + 1]) != np.sign(diff[idx]) else idx - 1
        neighbor = max(0, min(neighbor, len(diff) - 1))
        x0, x1 = diff[idx], diff[neighbor]
        if x1 == x0:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            eer_threshold = float(thresholds[idx])
        else:
            alpha = x0 / (x0 - x1)
            eer = float((1 - alpha) * (fpr[idx] + fnr[idx]) / 2.0 + alpha * (fpr[neighbor] + fnr[neighbor]) / 2.0)
            eer_threshold = float((1 - alpha) * thresholds[idx] + alpha * thresholds[neighbor])

    return eer, eer_threshold


def compute_roc_auc(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [np.ones_like(genuine_scores), np.zeros_like(impostor_scores)]
    )
    scores = np.concatenate([genuine_scores, impostor_scores])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Top-1 retrieval
# ---------------------------------------------------------------------------

def top1_retrieval_accuracy(
    query_embeddings: np.ndarray,
    query_true_cell_ids: Sequence[str],
    db: FingerprintDatabase,
) -> float:
    """For each query embedding, retrieve the nearest fingerprint (by
    cosine similarity) in the database and check whether it matches the
    query's true cell id (Algorithm 2, "Retrieval mode")."""
    fp_matrix, cell_order = db.matrix()
    if fp_matrix.shape[0] == 0:
        return float("nan")

    fp_norm = fp_matrix / (np.linalg.norm(fp_matrix, axis=1, keepdims=True) + 1e-12)
    q_norm = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    )
    sims = q_norm @ fp_norm.T  # (Q, N)
    best_idx = np.argmax(sims, axis=1)
    predictions = [cell_order[i] for i in best_idx]

    correct = sum(
        1 for pred, true in zip(predictions, query_true_cell_ids) if pred == true
    )
    return correct / len(query_true_cell_ids)


def top1_retrieval_predictions(
    query_embeddings: np.ndarray,
    db: FingerprintDatabase,
) -> Tuple[List[str], np.ndarray]:
    """Like :func:`top1_retrieval_accuracy` but returns the per-query
    predicted cell id and its best-match similarity score, for building
    a detailed per-query retrieval results table."""
    fp_matrix, cell_order = db.matrix()
    if fp_matrix.shape[0] == 0:
        return [], np.zeros((0,), dtype=np.float32)

    fp_norm = fp_matrix / (np.linalg.norm(fp_matrix, axis=1, keepdims=True) + 1e-12)
    q_norm = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    )
    sims = q_norm @ fp_norm.T  # (Q, N)
    best_idx = np.argmax(sims, axis=1)
    predictions = [cell_order[i] for i in best_idx]
    best_scores = sims[np.arange(len(best_idx)), best_idx]
    return predictions, best_scores
