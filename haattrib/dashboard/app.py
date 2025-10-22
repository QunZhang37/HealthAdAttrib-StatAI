import streamlit as st, json, os, pandas as pd
st.set_page_config(page_title="Healthcare Attribution", layout="wide")
st.title("Healthcare Multi-Channel Attribution Dashboard")

run_dir = st.text_input("Artifacts directory (outputs/run1)", "outputs/run1")
summary_path = os.path.join(run_dir, "summary.json")
if os.path.exists(summary_path):
    s = json.load(open(summary_path))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Metrics")
        st.json({"Logistic": s.get("logit_metrics"), "LSTM": s.get("lstm_metrics")})
    with col2:
        st.subheader("Channel Contributions (Markov)")
        st.image(os.path.join(run_dir, "markov_contrib.png"))
    st.subheader("Channel Contributions (Shapley)")
    st.image(os.path.join(run_dir, "shapley_contrib.png"))
else:
    st.info("Run the pipeline first to populate artifacts.")
