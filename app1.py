import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
import random

# ====================================================
# CONFIG
# ====================================================

st.set_page_config(
    page_title="OCR + NLP Certificados Médicos",
    layout="wide"
)

st.title("📄 OCR + NLP de Certificados Médicos")

# ====================================================
# NLTK
# ====================================================

@st.cache_resource
def descargar_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

descargar_nltk()

# ====================================================
# OCR
# ====================================================

@st.cache_resource
def cargar_ocr():
    return easyocr.Reader(['es', 'en'], gpu=False)

reader = cargar_ocr()

# ====================================================
# STOPWORDS
# ====================================================

STOPWORDS_ES = set(stopwords.words('spanish'))

STOPWORDS_DOMINIO = {
    'sr', 'sra', 'dr', 'dra',
    'certificado', 'fecha',
    'medico', 'médico',
    'dias', 'día', 'días',
    'lima', 'piura',
    'trujillo', 'arequipa'
}

STOPWORDS_COMPLETO = STOPWORDS_ES | STOPWORDS_DOMINIO

# ====================================================
# FUNCIONES NLP
# ====================================================

def convertir_minusculas(texto):
    if pd.isna(texto):
        return ''
    return texto.lower()

def eliminar_caracteres_especiales(texto):

    if not texto:
        return ''

    texto = re.sub(
        r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s.,:/\-]',
        ' ',
        texto
    )

    return texto

def normalizar_espacios(texto):

    if not texto:
        return ''

    texto = re.sub(r'[\n\r\t]+', ' ', texto)
    texto = re.sub(r' {2,}', ' ', texto)

    return texto.strip()

def tokenizar(texto):

    if not texto:
        return []

    return word_tokenize(texto, language='spanish')

def eliminar_stopwords(tokens):

    resultado = [
        t for t in tokens
        if t.lower() not in STOPWORDS_COMPLETO
        and len(t) > 1
        and not re.fullmatch(r'\d+', t)
        and not re.fullmatch(r'[.,:/\-]+', t)
    ]

    return resultado

def pipeline_limpieza(texto):

    t1 = convertir_minusculas(texto)

    t2 = eliminar_caracteres_especiales(t1)

    t3 = normalizar_espacios(t2)

    tokens = tokenizar(t3)

    tokens_limpios = eliminar_stopwords(tokens)

    texto_final = ' '.join(tokens_limpios)

    return {
        'texto_minusculas': t1,
        'texto_sin_especiales': t2,
        'texto_normalizado': t3,
        'tokens': tokens,
        'tokens_limpios': tokens_limpios,
        'texto_preprocesado': texto_final,
        'n_tokens_original': len(tokens),
        'n_tokens_limpios': len(tokens_limpios)
    }

# ====================================================
# SUBIR IMÁGENES
# ====================================================

uploaded_files = st.file_uploader(
    "📤 Subir certificados médicos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ====================================================
# MOSTRAR IMÁGENES
# ====================================================

if uploaded_files:

    st.subheader("🖼️ Vista previa")

    cols = st.columns(3)

    for i, file in enumerate(uploaded_files[:6]):

        img = Image.open(file)

        with cols[i % 3]:
            st.image(img, caption=file.name)

# ====================================================
# PROCESAR OCR
# ====================================================

if uploaded_files:

    if st.button("🚀 Ejecutar OCR + NLP"):

        resultados_finales = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            try:

                image = Image.open(uploaded_file)

                resultado_ocr = reader.readtext(
                    np.array(image),
                    detail=0,
                    paragraph=True
                )

                texto_ocr = "\n".join(resultado_ocr)

            except Exception as e:

                texto_ocr = f"ERROR: {str(e)}"

            # NLP
            resultado_nlp = pipeline_limpieza(texto_ocr)

            resultados_finales.append({

                'archivo': uploaded_file.name,

                'texto_ocr': texto_ocr,

                'texto_preprocesado': resultado_nlp['texto_preprocesado'],

                'tokens_originales': resultado_nlp['n_tokens_original'],

                'tokens_limpios': resultado_nlp['n_tokens_limpios'],

                'reduccion_tokens': round(
                    (
                        resultado_nlp['n_tokens_original']
                        - resultado_nlp['n_tokens_limpios']
                    )
                    /
                    max(resultado_nlp['n_tokens_original'], 1)
                    * 100,
                    2
                )
            })

            progress.progress((i + 1) / len(uploaded_files))

        # ====================================================
        # DATAFRAME
        # ====================================================

        df_final = pd.DataFrame(resultados_finales)

        st.success("✅ OCR FINALIZADO")

        st.subheader("📊 Resultados")

        st.dataframe(df_final)

        # ====================================================
        # MÉTRICAS
        # ====================================================

        st.subheader("📈 Métricas")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tokens Originales",
            round(df_final['tokens_originales'].mean(), 1)
        )

        c2.metric(
            "Tokens Limpios",
            round(df_final['tokens_limpios'].mean(), 1)
        )

        c3.metric(
            "Reducción %",
            round(df_final['reduccion_tokens'].mean(), 1)
        )

        # ====================================================
        # TOP PALABRAS
        # ====================================================

        st.subheader("📌 Top Palabras")

        texto_total = " ".join(df_final['texto_preprocesado'])

        palabras = texto_total.split()

        frecuencias = Counter(palabras)

        top30 = frecuencias.most_common(20)

        if len(top30) > 0:

            palabras_top, conteos = zip(*top30)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.barh(
                list(reversed(palabras_top)),
                list(reversed(conteos))
            )

            st.pyplot(fig)

        # ====================================================
        # COMPARACIÓN
        # ====================================================

        st.subheader("🔍 Comparación OCR vs Limpieza")

        idx = st.selectbox(
            "Seleccionar certificado",
            range(len(df_final)),
            format_func=lambda x: df_final.iloc[x]['archivo']
        )

        st.markdown("### Texto OCR")

        st.write(df_final.iloc[idx]['texto_ocr'])

        st.markdown("### Texto Limpio")

        st.write(df_final.iloc[idx]['texto_preprocesado'])

        # ====================================================
        # DESCARGAS
        # ====================================================

        csv = df_final.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            "⬇️ Descargar CSV",
            csv,
            "resultado_ocr.csv",
            "text/csv"
        )

        from io import BytesIO

        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_final.to_excel(writer, index=False)

        st.download_button(
            "⬇️ Descargar Excel",
            output.getvalue(),
            "resultado_ocr.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )