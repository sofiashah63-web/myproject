"""Train the disease prediction model and save it to models/disease_model.pkl.

Usage:  python train_model.py
"""
from src.config import MODEL_PATH
from src.data_utils import load_disease_symptoms
from src.model import save, train


def main() -> None:
    disease_df = load_disease_symptoms()
    n_symptoms = len({s for row in disease_df["symptoms"] for s in row})
    print(f"Loaded {len(disease_df)} diseases and {n_symptoms} unique symptoms.")

    predictor, metrics = train(disease_df)
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print("(sanity check on simulated partial inputs, not clinical accuracy)")

    save(predictor)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
