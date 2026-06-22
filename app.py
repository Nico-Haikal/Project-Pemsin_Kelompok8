import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict student academic performance using Machine Learning")

st.markdown("---")

# ===== Identitas Mahasiswa =====

st.subheader("Identitas Mahasiswa")

id_col1, id_col2 = st.columns(2)

with id_col1:
    Nama = st.text_input(
        "Nama",
        placeholder="Masukkan nama lengkap"
    )

with id_col2:
    NIM = st.text_input(
        "NIM",
        placeholder="Masukkan NIM"
    )

st.markdown("---")

# ===== Input Fitur Prediksi =====

st.subheader("Data Akademik")

col1, col2 = st.columns(2)

with col1:

    Hours_Studied = st.slider(
        "Hours Studied",
        1, 45, 20
    )

    Attendance = st.slider(
        "Attendance (%)",
        50, 100, 80
    )

    Sleep_Hours = st.slider(
        "Sleep Hours",
        4, 10, 7
    )

    Previous_Scores = st.slider(
        "Previous Scores",
        40, 100, 70
    )

    Tutoring_Sessions = st.slider(
        "Tutoring Sessions",
        0, 10, 3
    )

with col2:

    Motivation_Level = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

    Internet_Access = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

# Encoding

motivation_map = {
    "Low":0,
    "Medium":1,
    "High":2
}

internet_map = {
    "No":0,
    "Yes":1
}

gender_map = {
    "Female":0,
    "Male":1
}

Motivation_Level_enc = motivation_map[Motivation_Level]
Internet_Access_enc = internet_map[Internet_Access]
Gender_enc = gender_map[Gender]

# Nilai default untuk fitur lain

Parental_Involvement = 1
Access_to_Resources = 1
Extracurricular_Activities = 1
Family_Income = 1
Teacher_Quality = 1
School_Type = 0
Peer_Influence = 1
Physical_Activity = 3
Learning_Disabilities = 0
Parental_Education_Level = 1
Distance_from_Home = 1

st.markdown("---")

if st.button("Predict Performance"):

    # Validasi identitas mahasiswa wajib diisi dulu
    if not Nama.strip() or not NIM.strip():
        st.warning("⚠️ Mohon isi Nama dan NIM terlebih dahulu sebelum melakukan prediksi.")
    else:

        data = np.array([[
            Hours_Studied,
            Attendance,
            Parental_Involvement,
            Access_to_Resources,
            Extracurricular_Activities,
            Sleep_Hours,
            Previous_Scores,
            Motivation_Level_enc,
            Internet_Access_enc,
            Tutoring_Sessions,
            Family_Income,
            Teacher_Quality,
            School_Type,
            Peer_Influence,
            Physical_Activity,
            Learning_Disabilities,
            Parental_Education_Level,
            Distance_from_Home,
            Gender_enc
        ]])

        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0]

        probability = model.predict_proba(data_scaled)[0]

        poor_prob = probability[0] * 100
        good_prob = probability[1] * 100

        # Clamp ke range 0-100 biar aman dari floating point error
        poor_prob_clamped = max(0, min(100, int(round(poor_prob))))
        good_prob_clamped = max(0, min(100, int(round(good_prob))))

        st.subheader("Prediction Result")

        # ===== Tampilkan Identitas + Hasil Prediksi =====
        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Nama", Nama)

        with info_col2:
            st.metric("NIM", NIM)

        with info_col3:
            if prediction == 1:
                st.metric("Hasil Prediksi", "Good Performance")
            else:
                st.metric("Hasil Prediksi", "Poor Performance")

        st.markdown("---")

        if prediction == 1:

            st.success(f"🎓 {Nama} ({NIM}) diprediksi memiliki **Good Performance**")

            st.metric(
                "Confidence",
                f"{good_prob:.2f}%"
            )

            st.progress(good_prob_clamped)

        else:

            st.error(f"⚠️ {Nama} ({NIM}) diprediksi memiliki **Poor Performance**")

            st.metric(
                "Confidence",
                f"{poor_prob:.2f}%"
            )

            st.progress(poor_prob_clamped)

        st.markdown("---")

        st.subheader("Prediction Probability")

        st.write(
            f"📉 Poor Performance : {poor_prob:.2f}%"
        )

        st.write(
            f"📈 Good Performance : {good_prob:.2f}%"
        )

        st.progress(good_prob_clamped)

        with st.expander("Debug Information"):

            st.write("Nama:", Nama)
            st.write("NIM:", NIM)

            st.write(
                "Raw Prediction:",
                prediction
            )

            st.write(
                "Raw Probability:",
                probability
            )
