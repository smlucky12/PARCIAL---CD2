
# ============================================================
# APP STREAMLIT — OCR + NLP DE CERTIFICADOS MÉDICOS PERUANOS
# ============================================================
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import easyocr
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image
import tempfile
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="OCR + NLP Certificados Médicos",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# DESCARGAR RECURSOS NLTK
# ============================================================

nltk.download('punkt')
nltk.download('stopwords')

# ============================================================
# TÍTULO
# ============================================================

st.title("🏥 Sistema Inteligente OCR + NLP")
st.subheader("Análisis Automático de Certificados Médicos Peruanos")

st.markdown("""
Esta aplicación permite:

✅ Cargar imágenes de certificados médicos

✅ Realizar OCR automáticamente

✅ Limpiar y preprocesar texto

✅ Aplicar técnicas NLP

✅ Extraer palabras clave

✅ Generar nube de palabras

✅ Mostrar métricas

✅ Descargar resultados

""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuración")

idioma = st.sidebar.selectbox(
    "Idioma OCR",
    ["Español + Inglés", "Solo Español"]
)

mostrar_preprocesamiento = st.sidebar.checkbox(
    "Mostrar imágenes preprocesadas",
    value=True
)

# ============================================================
# OCR
# ============================================================

@st.cache_resource

def cargar_ocr():

    if idioma == "Solo Español":
        return easyocr.Reader(['es'], gpu=False)

    return easyocr.Reader(['es', 'en'], gpu=False)

reader = cargar_ocr()

# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS_ES = set(stopwords.words('spanish'))

STOPWORDS_DOMINIO = {
    'sr', 'sra', 'dr', 'dra', 'doctor', 'doctora',
    'certificado', 'certifica', 'medico', 'médico',
    'fecha', 'dias', 'día', 'días', 'paciente',
    'lima', 'peru', 'perú', 'hospital', 'clinica',
    'clínica', 'descanso', 'solicitud', 'presente',
    'inicio', 'fin', 'edad', 'sexo', 'dni', 'nhc'
}

STOPWORDS_COMPLETO = STOPWORDS_ES | STOPWORDS_DOMINIO

# ============================================================
# FUNCIONES DE PREPROCESAMIENTO DE IMAGEN
# ============================================================


def preprocesar_imagen(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionamiento
    gray = cv2.resize(gray, None, fx=2, fy=2)

    # Reducción de ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mejora de contraste
    contrast = cv2.equalizeHist(blur)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return {
        'original': img,
        'gray': gray,
        'blur': blur,
        'contrast': contrast,
        'final': thresh
    }

# ============================================================
# FUNCIONES NLP
# ============================================================


def convertir_minusculas(texto):

    if pd.isna(texto):
        return ''

    return texto.lower()



def eliminar_caracteres_especiales(texto):

    texto = re.sub(
        r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s.,:/\-]',
        ' ',
        texto
    )

    return texto



def normalizar_espacios(texto):

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
        'minusculas': t1,
        'sin_especiales': t2,
        'normalizado': t3,
        'tokens': tokens,
        'tokens_limpios': tokens_limpios,
        'texto_final': texto_final,
        'n_tokens_original': len(tokens),
        'n_tokens_limpios': len(tokens_limpios)
    }

# ============================================================
# CLASIFICACIÓN SIMPLE
# ============================================================


def clasificar_documento(texto):

    texto = texto.lower()

    categorias = {
        'Traumatología': [
            'fractura', 'trauma', 'luxacion',
            'esguince', 'golpe', 'lesion'
        ],

        'Respiratorio': [
            'gripe', 'covid', 'tos',
            'fiebre', 'bronquitis', 'pulmon'
        ],

        'Administrativo': [
            'descanso', 'certificado',
            'incapacidad', 'reposo'
        ]
    }

    puntajes = {}

    for categoria, palabras in categorias.items():

        score = sum(
            1 for palabra in palabras
            if palabra in texto
        )

        puntajes[categoria] = score

    mejor_categoria = max(puntajes, key=puntajes.get)

    return mejor_categoria

# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📤 Subir imágenes de certificados",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# ============================================================
# PROCESAMIENTO
# ============================================================

if uploaded_files:

    resultados_finales = []

    st.success(f"✅ {len(uploaded_files)} imágenes cargadas")

    for uploaded_file in uploaded_files:

        st.markdown("---")

        st.header(f"📄 {uploaded_file.name}")

        # ====================================================
        # GUARDAR TEMPORAL
        # ====================================================

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # ====================================================
        # MOSTRAR ORIGINAL
        # ====================================================

        image = Image.open(tfile.name)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(image, use_container_width=True)

        # ====================================================
        # PREPROCESAMIENTO
        # ====================================================

        pre = preprocesar_imagen(tfile.name)

        with col2:

            if mostrar_preprocesamiento:

                st.subheader("Imagen Preprocesada")
                st.image(pre['final'], use_container_width=True)

        # ====================================================
        # OCR
        # ====================================================

        with st.spinner("Procesando OCR..."):

            resultado_ocr = reader.readtext(
                pre['final'],
                detail=0,
                paragraph=True
            )

            texto_ocr = "\n".join(resultado_ocr)

        st.subheader("📝 Texto Extraído (OCR)")

        st.text_area(
            "Texto OCR",
            texto_ocr,
            height=200
        )

        # ====================================================
        # NLP
        # ====================================================

        resultado_nlp = pipeline_limpieza(texto_ocr)

        texto_limpio = resultado_nlp['texto_final']

        st.subheader("🧹 Texto Preprocesado")

        st.text_area(
            "Texto limpio",
            texto_limpio,
            height=150
        )

        # ====================================================
        # MÉTRICAS
        # ====================================================

        st.subheader("📊 Métricas")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Tokens Originales",
                resultado_nlp['n_tokens_original']
            )

        with m2:
            st.metric(
                "Tokens Limpios",
                resultado_nlp['n_tokens_limpios']
            )

        with m3:

            if resultado_nlp['n_tokens_original'] > 0:

                reduccion = (
                    (
                        resultado_nlp['n_tokens_original']
                        - resultado_nlp['n_tokens_limpios']
                    )
                    /
                    resultado_nlp['n_tokens_original']
                ) * 100

            else:
                reduccion = 0

            st.metric(
                "Reducción",
                f"{reduccion:.1f}%"
            )

        # ====================================================
        # PALABRAS CLAVE
        # ====================================================

        st.subheader("🔑 Palabras Clave")

        frecuencias = Counter(resultado_nlp['tokens_limpios'])

        top_palabras = frecuencias.most_common(10)

        df_keywords = pd.DataFrame(
            top_palabras,
            columns=['Palabra', 'Frecuencia']
        )

        st.dataframe(df_keywords, use_container_width=True)

        # ====================================================
        # CLASIFICACIÓN
        # ====================================================

        categoria = clasificar_documento(texto_limpio)

        st.subheader("🏷️ Clasificación del Documento")

        st.success(f"Categoría detectada: {categoria}")

        # ====================================================
        # NUBE DE PALABRAS
        # ====================================================

        if len(texto_limpio) > 10:

            st.subheader("☁️ Nube de Palabras")

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color='white'
            ).generate(texto_limpio)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wordcloud)
            ax.axis('off')

            st.pyplot(fig)

        # ====================================================
        # GRÁFICO DE FRECUENCIAS
        # ====================================================

        st.subheader("📈 Top Palabras Frecuentes")

        top10 = frecuencias.most_common(10)

        palabras = [x[0] for x in top10]
        valores = [x[1] for x in top10]

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.barh(palabras, valores)

        ax2.set_title('Top 10 Palabras')
        ax2.set_xlabel('Frecuencia')

        st.pyplot(fig2)

        # ====================================================
        # RESULTADOS
        # ====================================================

        resultados_finales.append({
            'archivo': uploaded_file.name,
            'texto_ocr': texto_ocr,
            'texto_limpio': texto_limpio,
            'categoria': categoria,
            'tokens_originales': resultado_nlp['n_tokens_original'],
            'tokens_limpios': resultado_nlp['n_tokens_limpios']
        })

        # ====================================================
        # ELIMINAR ARCHIVO TEMPORAL
        # ====================================================

        try:
            tfile.close()
            os.remove(tfile.name)

        except:
            pass

    # ========================================================
    # EXPORTACIÓN FINAL
    # ========================================================

    st.markdown('---')

    st.header('📥 Exportar Resultados')

    df_final = pd.DataFrame(resultados_finales)

    st.dataframe(df_final, use_container_width=True)

    csv = df_final.to_csv(index=False).encode('utf-8-sig')

    st.download_button(
        label='⬇️ Descargar CSV',
        data=csv,
        file_name='resultados_ocr_nlp.csv',
        mime='text/csv'
    )
