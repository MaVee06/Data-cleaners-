import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Water Quality — Irrigation Suitability", page_icon="💧", layout="wide")

FEATURES = ["PH", "dissolved_oxgen_ppm", "Electrical Conductivity"]

st.title("💧 Irrigation Suitability Predictor")
st.caption(
    "Random Forest Classifier trained on citizen-science water quality data "
    "from the Limpopo Biosphere Reserves (Kruger 2 Canyons, Vhembe, Marico)."
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
st.sidebar.header("1. Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload the water quality CSV", type=["csv"],
    help="citizen-science-water-quality-monitoring-wq-parameters_local_time.csv"
)

DEFAULT_PATH = "citizen-science-water-quality-monitoring-wq-parameters_local_time (1).csv"


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes_or_path):
    df = pd.read_csv(file_bytes_or_path)

    # Match the cleaning steps used to build the model in the notebook
    df.drop_duplicates(inplace=True)
    drop_cols = [c for c in ["Biosphere", "id", "other_weather", "start_comment"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    df.dropna(subset=["Irrigation Suitability"], inplace=True)

    numeric_cols = [c for c in ["PH", "dissolved_oxgen_ppm", "Electrical Conductivity",
                                 "Air Temperature", "dissolved_oxygen_percent"] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    zero_cols = [c for c in ["Nitrate", "Phosphate", "tds"] if c in df.columns]
    df.drop(columns=zero_cols, inplace=True)

    df = df[(df["PH"] >= 0) & (df["PH"] <= 14)]

    df["suitable"] = df["Irrigation Suitability"].str.contains("NOT").apply(lambda x: 0 if x else 1)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df):
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["suitable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, report, cm


data_source = uploaded_file if uploaded_file is not None else DEFAULT_PATH

try:
    df_clean = load_and_clean(data_source)
    model_rf, report, cm = train_model(df_clean)
    data_ready = True
except FileNotFoundError:
    data_ready = False
    st.warning(
        "No dataset found. Upload the water quality CSV in the sidebar to train the model "
        "(the app looks for it locally otherwise)."
    )
except Exception as e:
    data_ready = False
    st.error(f"Couldn't load/clean the dataset: {e}")

# ---------------------------------------------------------------------------
# Prediction UI
# ---------------------------------------------------------------------------
st.sidebar.header("2. Water sample readings")
ph = st.sidebar.slider("pH", min_value=0.0, max_value=14.0, value=8.0, step=0.1)
do_ppm = st.sidebar.number_input("Dissolved Oxygen (ppm)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
ec = st.sidebar.number_input("Electrical Conductivity", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)

predict_clicked = st.sidebar.button("Predict suitability", type="primary", disabled=not data_ready)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    if not data_ready:
        st.info("Load a dataset to enable predictions.")
    elif predict_clicked:
        sample = pd.DataFrame([[ph, do_ppm, ec]], columns=FEATURES)
        pred = model_rf.predict(sample)[0]
        proba = model_rf.predict_proba(sample)[0]

        if pred == 1:
            st.success(f"✅ Suitable for irrigation  (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"⚠️ Not suitable for irrigation  (confidence: {proba[0]*100:.1f}%)")

        st.progress(float(proba[1]), text=f"P(Suitable) = {proba[1]*100:.1f}%")
    else:
        st.write("Set the readings in the sidebar and click **Predict suitability**.")

with col2:
    st.subheader("Feature importance")
    if data_ready:
        importances = pd.Series(model_rf.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(importances)
    else:
        st.write("—")

st.divider()

st.subheader("Model performance (held-out test set)")
if data_ready:
    c1, c2, c3 = st.columns(3)
    c1.metric("F1 — Not Suitable", f"{report['Not Suitable']['f1-score']:.2f}")
    c2.metric("F1 — Suitable", f"{report['Suitable']['f1-score']:.2f}")
    c3.metric("Accuracy", f"{report['accuracy']:.2f}")

    with st.expander("Full classification report"):
        st.dataframe(pd.DataFrame(report).transpose())

    with st.expander("Confusion matrix"):
        st.dataframe(pd.DataFrame(cm, index=["Actual: Not Suitable", "Actual: Suitable"],
                                   columns=["Pred: Not Suitable", "Pred: Suitable"]))
else:
    st.write("Metrics will appear once a dataset is loaded.")
