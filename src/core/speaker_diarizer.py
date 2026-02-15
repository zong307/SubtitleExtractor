"""Optional speaker diarization using CAM++ via funasr."""

from __future__ import annotations

import string
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from src.models.datatypes import SpeechSegment


class SpeakerDiarizer:
    """Identify speakers in audio segments using CAM++ embeddings + clustering."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the funasr CAM++ model."""
        if self._model is None:
            from funasr import AutoModel

            logger.info("Loading CAM++ speaker embedding model...")
            self._model = AutoModel(model="cam++")
            logger.info("CAM++ model loaded.")

    def identify_speakers(
        self,
        audio_path: str,
        segments: list[SpeechSegment],
        num_speakers: Optional[int] = None,
    ) -> dict[int, str]:
        """Extract embeddings per segment and cluster into speaker identities.

        Args:
            audio_path: path to the full audio file (16 kHz mono WAV).
            segments: speech segments (matching the order of transcription segments).
            num_speakers: if provided, force this many clusters; otherwise auto-detect.

        Returns:
            Mapping from segment index to speaker label ("A", "B", "C", ...).
        """
        self._ensure_model()

        if not segments:
            return {}

        logger.info(f"Extracting speaker embeddings for {len(segments)} segment(s)...")
        full_audio, sr = sf.read(audio_path, dtype="float32")

        embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for idx, seg in enumerate(segments):
            start_sample = int(seg.start_time * sr)
            end_sample = int(seg.end_time * sr)
            chunk = full_audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:
                # Skip very short segments (< 0.1s)
                continue

            emb = self._extract_embedding(chunk, sr)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)

        if len(embeddings) < 2:
            # Not enough segments to cluster – assign all to speaker "A"
            logger.warning(
                "Too few segments for speaker clustering. Assigning all to Speaker A."
            )
            return {i: "A" for i in range(len(segments))}

        embedding_matrix = np.vstack(embeddings)

        labels = self._cluster(
            embedding_matrix,
            num_speakers=num_speakers,
        )

        # Map cluster labels to letters A, B, C, ...
        label_letters = list(string.ascii_uppercase)
        unique_labels = sorted(set(labels))
        label_map = {
            lbl: label_letters[i % len(label_letters)]
            for i, lbl in enumerate(unique_labels)
        }

        result: dict[int, str] = {}
        for valid_idx, seg_idx in enumerate(valid_indices):
            result[seg_idx] = label_map[labels[valid_idx]]

        # For segments that were skipped (too short), assign the nearest speaker
        for idx in range(len(segments)):
            if idx not in result:
                result[idx] = self._nearest_speaker(idx, result)

        n_speakers = len(set(result.values()))
        logger.info(f"Speaker diarization complete: {n_speakers} speaker(s) detected.")
        return result

    def _extract_embedding(
        self, audio_chunk: np.ndarray, sr: int
    ) -> Optional[np.ndarray]:
        """Extract a speaker embedding for a single audio chunk."""
        try:
            # Write chunk to a temporary WAV file for funasr
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio_chunk, sr, subtype="PCM_16")

            res = self._model.generate(input=tmp_path)

            Path(tmp_path).unlink(missing_ok=True)

            if res and len(res) > 0:
                emb = res[0].get("spk_embedding", None)
                if emb is None:
                    emb = res[0].get("embedding", None)
                if emb is not None:
                    return np.array(emb, dtype=np.float32).flatten()
        except Exception as e:
            logger.warning(f"Embedding extraction failed for a segment: {e}")
        return None

    @staticmethod
    def _cluster(
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> list[int]:
        """Cluster speaker embeddings using Agglomerative Clustering."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        if num_speakers is not None and num_speakers > 0:
            model = AgglomerativeClustering(
                n_clusters=num_speakers, metric="cosine", linkage="average"
            )
            labels = model.fit_predict(embeddings)
            return labels.tolist()

        # Auto-detect number of speakers: try 2..min(10, n_samples) and pick best silhouette
        max_k = min(10, len(embeddings))
        if max_k < 2:
            return [0] * len(embeddings)

        best_k = 2
        best_score = -1.0
        for k in range(2, max_k + 1):
            model = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            )
            labels = model.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k

        model = AgglomerativeClustering(
            n_clusters=best_k, metric="cosine", linkage="average"
        )
        labels = model.fit_predict(embeddings)
        return labels.tolist()

    @staticmethod
    def _nearest_speaker(idx: int, speaker_map: dict[int, str]) -> str:
        """For a missing index, return the speaker of the nearest known segment."""
        if not speaker_map:
            return "A"
        nearest = min(speaker_map.keys(), key=lambda k: abs(k - idx))
        return speaker_map[nearest]
