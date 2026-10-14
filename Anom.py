import pandas as pd
import joblib
import streamlit as st

model = joblib.load("anomaly_model.joblib")      # hakuna retraining


def anomaly_page():
    st.title("Anomaly Detection")

    # ---------- Form 1: Manual Entry ----------
    st.header("1. Manual Entry Form")
    with st.form("manual_form"):
        kiasi = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        tuma1 = st.form_submit_button("Angalia")
    if tuma1:
        jibu = model.predict(pd.DataFrame({"Transaction_Amount": [kiasi]}))[0]
        if jibu == 1:
            st.success(f"{kiasi:,.2f} ni Normal")
        else:
            st.error(f"{kiasi:,.2f} ni Anomalous")

    # ---------- Form 2: Upload ----------
    st.header("2. Upload Form")
    with st.form("upload_form"):
        faili = st.file_uploader("Pakia CSV au Excel", type=["csv", "xlsx"])
        tuma2 = st.form_submit_button("Changanua")
    if tuma2:
        if faili is None:
            st.warning("Tafadhali pakia faili kwanza.")
        else:
            data = pd.read_csv(faili) if faili.name.endswith(".csv") else pd.read_excel(faili)
            data = data.rename(columns={"amount": "Transaction_Amount"})
            if "Transaction_Amount" not in data.columns:
                st.error("Faili lazima liwe na column 'amount' au 'Transaction_Amount'.")
            else:
                data["result"] = model.predict(data[["Transaction_Amount"]])
                data["status"] = data["result"].map({1: "Normal", -1: "Anomalous"})
                st.write("Normal:", (data["result"] == 1).sum(),
                         "| Anomalous:", (data["result"] == -1).sum())
                st.dataframe(data[data["status"] == "Anomalous"])


# ---------- Sidebar navigation ----------
if __name__ == "__main__":
    ukurasa = st.sidebar.radio("Menu", ["Home", "Anomaly Detection"])
    if ukurasa == "Anomaly Detection":
        anomaly_page()
    else:
        st.title("Dashboard")