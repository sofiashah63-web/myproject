# Disease Prediction using Symptoms

A Streamlit app where you pick your symptoms and a machine learning model
(scikit-learn logistic regression) predicts the most likely diseases.

## Project structure

```
disease_prediction/
├── app.py              # Streamlit interface
├── train_model.py      # trains and saves the model
├── data/dataset.csv    # disease -> symptoms dataset
├── models/             # saved model (created by train_model.py)
├── src/
│   ├── config.py       # paths and constants
│   ├── data_utils.py   # loading and cleaning the dataset
│   └── model.py        # training, saving, prediction
├── .streamlit/config.toml   # theme
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python train_model.py      # optional: the app trains automatically if no model exists
streamlit run app.py
```

Then open http://localhost:8501.

## How it works

1. Each disease's symptoms are cleaned and merged (the CSV has duplicate rows).
2. Symptoms are one-hot encoded (`MultiLabelBinarizer`).
3. The dataset has about one row per disease, so training samples are created by
   randomly keeping 40-100% of each disease's symptoms. This teaches the model
   to handle partial input.
4. A `LogisticRegression` model is trained, and the app shows the top 3 diseases
   with confidence and which of your symptoms matched.

## Limitations

- The dataset is small (84 diseases) and some symptom entries are messy
  (e.g. "red", "loose", "from manito depression"). Cleaning `data/dataset.csv`
  will improve the results.
- The accuracy printed by `train_model.py` is measured on simulated data from the
  same dataset, so it is only a sanity check, not real-world accuracy.
- Not a medical tool. Always consult a doctor.
