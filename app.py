import streamlit as st
from asr import record_and_predict, spell_check

st.header("Transcription vocale - FormaScience")

st.subheader("Essayer le modèle en temps réel")

st.sidebar.title("Paramètres")
duration = st.sidebar.slider("Durée de l'enregistrement", 0.0, 10.0, 5.0)

if st.button("Commencer l'enregistrement"):
    with st.spinner("Recording..."):
        prediction = record_and_predict(duration=duration)
        st.write("**Prediction**: ", prediction[0])
        st.write("**Spell Check**: ", spell_check(prediction[0]))