"""Disease Prediction using Symptoms - Streamlit app.

Run with:  streamlit run app.py
"""
import streamlit as st

from src import model as model_lib
from src.config import MODEL_PATH, TOP_K
from src.data_utils import load_disease_symptoms, pretty

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> model_lib.DiseasePredictor:
    """Load the saved model, training it first if it does not exist yet."""
    if not MODEL_PATH.exists():
        predictor, _ = model_lib.train(load_disease_symptoms())
        model_lib.save(predictor)
        return predictor
    return model_lib.load()


def render_prediction(rank: int, result: dict) -> None:
    """Show one predicted disease with its confidence and symptom overlap."""
    name = pretty(result["disease"])
    percent = result["probability"] * 100

    if rank == 0:
        st.success(f"### Most likely: {name}")
    else:
        st.markdown(f"**{rank + 1}. {name}**")
    st.progress(min(result["probability"], 1.0), text=f"Confidence: {percent:.1f}%")

    with st.expander("Symptom match details", expanded=(rank == 0)):
        matched = ", ".join(pretty(s) for s in result["matched"]) or "None"
        missing = ", ".join(pretty(s) for s in result["missing"]) or "None"
        st.markdown(f"**Matched symptoms:** {matched}")
        st.markdown(f"**Other symptoms of this disease:** {missing}")


def main() -> None:
    predictor = get_predictor()

    st.title("Disease Prediction using Symptoms")
    st.caption("Select the symptoms you are experiencing and the model will suggest likely conditions.")
    
    selected = st.multiselect(
        "Symptoms",
        options=predictor.symptoms,
        format_func=pretty,
        placeholder="Search and select one or more symptoms",
    )

    if selected:
        st.write(f"{len(selected)} symptom(s) selected")

    if st.button("Predict Disease", type="primary", use_container_width=True):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            st.divider()
            st.subheader("Prediction")
            for rank, result in enumerate(predictor.predict_top_k(selected, k=TOP_K)):
                render_prediction(rank, result)

    st.divider()
    st.info(
        "This tool is for educational purposes only and is **not** a medical diagnosis. "
        "Please consult a qualified doctor for any health concern."
    )


main()
