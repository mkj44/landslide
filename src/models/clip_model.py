"""CLIP ViT-B/32 Zero-Shot and Linear Probe evaluation for satellite landslide detection.

NOTE ON SPECTRAL LIMITATIONS:
Pretrained vision-language models such as OpenAI CLIP ViT-B/32 were trained strictly
on 3-channel (RGB) natural photography. Consequently, CLIP is RGB-only; it does not
support native Sentinel-2 red-edge (B5-B7), short-wave infrared (B11-B12), or topographic
layers (DEM, Slope). Multi-spectral patches are projected to RGB (B4, B3, B2) for CLIP evaluation.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from PIL import Image

from src.utils.logging import get_logger
from src.evaluation.metrics import compute_classification_metrics

logger = get_logger("clip_model")

# Standard prompt pairs: (positive_prompt, negative_prompt)
CLIP_PROMPTS = [
    (
        "Template 1: Satellite scar vs Forest",
        "a satellite photo of a landslide scar with bare exposed earth",
        "a satellite photo of stable green forested mountain terrain",
    ),
    (
        "Template 2: Mountain slope with vs without landslide",
        "an aerial view of a steep mountain slope with a landslide",
        "an aerial view of a steep mountain slope without a landslide",
    ),
    (
        "Template 3: Soil erosion/mudslide vs Undisturbed vegetation",
        "satellite imagery showing soil erosion and mudslide damage",
        "satellite imagery showing normal undisturbed forest and vegetation",
    ),
    (
        "Template 4: Catastrophic landslide vs Stable terrain",
        "a high resolution satellite image of a catastrophic landslide",
        "a high resolution satellite image of stable mountain terrain",
    ),
]


class LandslideCLIP:
    """Zero-shot and Linear Probing wrapper for CLIP ViT-B/32 on RGB satellite imagery."""

    def __init__(self, device: Optional[torch.device] = None):
        import open_clip

        self.device = device or (torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu"))
        logger.info("Initializing CLIP ViT-B/32 on %s (RGB-only evaluation)...", self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @staticmethod
    def patch_to_rgb(img_14b: np.ndarray) -> Image.Image:
        """Converts a Landslide4Sense 14-band patch into a PIL RGB image.

        Sentinel-2 bands in L4S: B04 (Red: idx 3), B03 (Green: idx 2), B02 (Blue: idx 1).
        """
        rgb = np.stack([img_14b[:, :, 3], img_14b[:, :, 2], img_14b[:, :, 1]], axis=-1)
        # Robust 2nd-98th percentile scaling into uint8 [0, 255]
        p2, p98 = np.percentile(rgb, (2, 98))
        scaled = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1) * 255.0
        return Image.fromarray(scaled.astype(np.uint8))

    def encode_rgb_patches(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Extracts normalized 512-dim visual embeddings from a list of PIL RGB images."""
        tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(tensors)
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()

    def zero_shot_evaluation(
        self,
        pil_images: List[Image.Image],
        y_true: np.ndarray,
        temperature: float = 100.0,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates zero-shot classification performance across multiple prompt templates."""
        tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            img_embeds = F.normalize(self.model.encode_image(tensors), dim=-1)

        results = {}
        for name, pos_prompt, neg_prompt in CLIP_PROMPTS:
            text_tokens = self.tokenizer([pos_prompt, neg_prompt]).to(self.device)
            with torch.no_grad():
                text_embeds = F.normalize(self.model.encode_text(text_tokens), dim=-1) # (2, 512)
                # Cosine similarities
                sims = (img_embeds @ text_embeds.T) * (temperature / 100.0) # (N, 2)
                # Softmax probability for positive landslide class (col 0)
                probs = F.softmax(sims, dim=-1)[:, 0].cpu().numpy()

            metrics = compute_classification_metrics(y_true, probs, threshold=0.5)
            results[name] = metrics
            logger.info("CLIP Zero-Shot [%s]: F1=%.4f, Acc=%.4f, PR-AUC=%.4f",
                        name, metrics["f1"], metrics["accuracy"], metrics["pr_auc"])

        return results

    def fit_linear_probe(
        self,
        train_images: List[Image.Image],
        y_train: np.ndarray,
        val_images: List[Image.Image],
        y_val: np.ndarray,
        C: float = 1.0,
    ) -> Tuple[LogisticRegression, Dict[str, float], np.ndarray]:
        """Trains a regularized linear probe on top of frozen CLIP visual embeddings."""
        logger.info("Extracting CLIP embeddings for linear probe (Train: %d, Val: %d)...",
                    len(train_images), len(val_images))
        X_train = self.encode_rgb_patches(train_images)
        X_val = self.encode_rgb_patches(val_images)

        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        metrics = compute_classification_metrics(y_val, val_probs, threshold=0.5)
        logger.info("CLIP Linear Probe Validation: F1=%.4f, Acc=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f",
                    metrics["f1"], metrics["accuracy"], metrics["roc_auc"], metrics["pr_auc"])

        return clf, metrics, val_probs
