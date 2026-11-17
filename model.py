"""Model building, training, persistence and inference."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from config import (
    MIN_SYMPTOM_FRACTION,
    MODEL_PATH,
    RANDOM_STATE,
    SAMPLES_PER_DISEASE,
    TOP_K,
)


@dataclass
class DiseasePredictor:
    """Bundles the fitted encoders and classifier so they are saved together."""

    symptom_encoder: MultiLabelBinarizer
    label_encoder: LabelEncoder
    classifier: LogisticRegression
    disease_symptoms: dict[str, list[str]]

    @property
    def symptoms(self) -> list[str]:
        return list(self.symptom_encoder.classes_)

    def predict_top_k(self, selected: list[str], k: int = TOP_K) -> list[dict]:
        """Return the k most likely diseases with probability and symptom overlap."""
        vector = self.symptom_encoder.transform([selected])
        probabilities = self.classifier.predict_proba(vector)[0]
        top_idx = np.argsort(probabilities)[::-1][:k]

        results = []
        chosen = set(selected)
        for idx in top_idx:
            disease = self.label_encoder.inverse_transform([self.classifier.classes_[idx]])[0]
            known = set(self.disease_symptoms[disease])
            results.append(
                {
                    "disease": disease,
                    "probability": float(probabilities[idx]),
                    "matched": sorted(chosen & known),
                    "missing": sorted(known - chosen),
                }
            )
        return results


def _make_partial_samples(
    disease_df: pd.DataFrame, n_per_disease: int, rng: np.random.Generator
) -> tuple[list[list[str]], list[str]]:
    """Create synthetic samples where only a random part of the symptoms is present.

    The dataset has about one row per disease, so a model trained on it directly
    would only ever see complete symptom lists. Real users rarely know every
    symptom, so we simulate partial inputs.
    """
    samples, labels = [], []
    for disease, symptoms in zip(disease_df["disease"], disease_df["symptoms"]):
        for _ in range(n_per_disease):
            fraction = rng.uniform(MIN_SYMPTOM_FRACTION, 1.0)
            size = max(1, round(len(symptoms) * fraction))
            subset = rng.choice(symptoms, size=size, replace=False).tolist()
            samples.append(subset)
            labels.append(disease)
    return samples, labels


def train(disease_df: pd.DataFrame) -> tuple[DiseasePredictor, dict[str, float]]:
    """Train the model and return it with a quick sanity-check evaluation."""
    rng = np.random.default_rng(RANDOM_STATE)

    symptom_encoder = MultiLabelBinarizer().fit(disease_df["symptoms"])
    label_encoder = LabelEncoder().fit(disease_df["disease"])

    samples, labels = _make_partial_samples(disease_df, SAMPLES_PER_DISEASE, rng)
    X = symptom_encoder.transform(samples)
    y = label_encoder.transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    classifier = LogisticRegression(C=10, max_iter=3000)
    classifier.fit(X_train, y_train)

    proba = classifier.predict_proba(X_test)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    metrics = {
        "top1_accuracy": float(np.mean(classifier.classes_[proba.argmax(1)] == y_test)),
        "top3_accuracy": float(
            np.mean([t in classifier.classes_[row] for t, row in zip(y_test, top3)])
        ),
    }

    # Refit on everything so the final model has seen all synthetic samples.
    classifier.fit(X, y)

    predictor = DiseasePredictor(
        symptom_encoder=symptom_encoder,
        label_encoder=label_encoder,
        classifier=classifier,
        disease_symptoms=dict(zip(disease_df["disease"], disease_df["symptoms"])),
    )
    return predictor, metrics


def save(predictor: DiseasePredictor, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(predictor, f)


def load(path=MODEL_PATH) -> DiseasePredictor:
    with open(path, "rb") as f:
        return pickle.load(f)
