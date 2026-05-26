# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from io import BytesIO
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
import cv2
import gc
import seaborn as sns

from datetime import datetime

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="OCR + NLP Médico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('ggplot')

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f7fc;
}

/* TITULOS */

.titulo {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-top: 10px;
}

.subtitulo {
    font-size: 18px;
    color: #475569;
    text-align: center;
    margin-bottom: 30px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* METRICAS */

.metric-card {
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
}

.metric-title {
    font-size: 16px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 32px;
}

/* COLORES */

.card1 {
    background: linear-gradient(135deg,#2563eb,#60a5fa);
}

.card2 {
    background: linear-gradient(135deg,#9333ea,#c084fc);
}

.card3 {
    background: linear-gradient(135deg,#059669,#34d399);
}

.card4 {
    background: linear-gradient(135deg,#ea580c,#fb923c);
}

.card5 {
    background: linear-gradient(135deg,#dc2626,#f87171);
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {

    display: grid;

    grid-template-columns: repeat(6, 1fr);

    width: 100%;

    gap: 15px;

    margin-top: 20px;
}

.stTabs [data-baseweb="tab"] {

    height: 70px;

    width: 100%;

    border-radius: 18px;

    font-size: 18px;

    font-weight: 800;

    color: white;

    border: none;

    transition: 0.3s;
}

/* RESULTADOS */

button[id*="tab-0"] {

    background: linear-gradient(135deg,#2563eb,#60a5fa);
}

/* NLP */

button[id*="tab-1"] {

    background: linear-gradient(135deg,#9333ea,#c084fc);
}

/* WORDCLOUD */

button[id*="tab-2"] {

    background: linear-gradient(135deg,#059669,#34d399);
}

/* CLASIFICACION */

button[id*="tab-3"] {

    background: linear-gradient(135deg,#ea580c,#fb923c);
}

/* SENTIMIENTOS */

button[id*="tab-4"] {

    background: linear-gradient(135deg,#dc2626,#f87171);
}

/* EXPORTAR */

button[id*="tab-5"] {

    background: linear-gradient(135deg,#0f172a,#475569);
}

/* TAB ACTIVA */

.stTabs [aria-selected="true"] {

    transform: scale(1.05);

    box-shadow: 0px 8px 18px rgba(0,0,0,0.2);
}

/* BOTONES */

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#60a5fa);
    color: white;
    border-radius: 12px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stDownloadButton>button {
    background: linear-gradient(90deg,#059669,#34d399);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}

/* VISTA PREVIA */

.preview-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '''
    <div class="titulo">
    Aplicación de OCR y NLP para análisis automático de documentos en contexto peruano
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitulo">
    Extracción y análisis inteligente de texto en imágenes y documentos utilizando OCR y Procesamiento de Lenguaje Natural
    </div>
    ''',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Panel de Configuración")

st.sidebar.info("""
OCR automático  
Procesamiento NLP  
Clasificación temática  
WordCloud  
Resumen automático  
Análisis de sentimiento  
Extracción de entidades  
Exportación CSV y Excel  
""")

mostrar_preprocesamiento = st.sidebar.checkbox(
    "Mostrar Preprocesamiento",
    value=True
)

# =========================================================
# DESCARGAS NLTK
# =========================================================

@st.cache_resource
def descargar_nltk():

    nltk.download('punkt')
    nltk.download('stopwords')

descargar_nltk()

# =========================================================
# SPACY
# =========================================================

@st.cache_resource
def cargar_spacy():

    return spacy.blank("es")

nlp = cargar_spacy()

# =========================================================
# OCR
# =========================================================

@st.cache_resource
def cargar_ocr():

    return easyocr.Reader(
        ['es'],
        gpu=False,
        quantize=True
    )

reader = cargar_ocr()

# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS_ES = set(stopwords.words('spanish'))

STOPWORDS_DOMINIO = {

    'sr', 'sra', 'dr', 'dra',
    'certificado', 'certifica',
    'medico', 'médico',
    'fecha', 'dias', 'día',
    'lima', 'piura',
    'trujillo', 'arequipa',
    'escaneado', 'camscanner',
    'paciente', 'doctor',
    'doctora'

}

STOPWORDS_COMPLETO = (
    STOPWORDS_ES |
    STOPWORDS_DOMINIO
)

# =========================================================
# PREPROCESAMIENTO
# =========================================================

def preprocesar_imagen(pil_image):

    img = np.array(pil_image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    altura, ancho = img.shape[:2]

    max_dim = 1400

    if max(altura, ancho) > max_dim:

        escala = max_dim / max(altura, ancho)

        nuevo_ancho = int(ancho * escala)
        nueva_altura = int(altura * escala)

        img = cv2.resize(
            img,
            (nuevo_ancho, nueva_altura)
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return {

        'original': img,
        'gray': gray

    }

# =========================================================
# NLP
# =========================================================

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

    return word_tokenize(
        texto,
        language='spanish'
    )

def eliminar_stopwords(tokens):

    resultado = [

        t for t in tokens

        if t.lower() not in STOPWORDS_COMPLETO
        and len(t) > 2
        and not re.fullmatch(r'\d+', t)
        and t.isalpha()

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

        'texto_preprocesado': texto_final,

        'tokens': tokens_limpios,

        'n_tokens_original': len(tokens),

        'n_tokens_limpios': len(tokens_limpios)

    }

# =========================================================
# CLASIFICACION
# =========================================================

def clasificar_documento(texto):

    texto = texto.upper()

    categorias = {

        "Traumatología": [
            "TRAUMA",
            "FRACTURA",
            "ORTOPEDIA"
        ],

        "Neurología": [
            "NEURO",
            "CEREBRO",
            "MIGRAÑA"
        ],

        "Cardiología": [
            "CARDIO",
            "CORAZON",
            "PRESION"
        ],

        "Pediatría": [
            "PEDIATR",
            "NIÑO",
            "INFANTE"
        ]

    }

    for categoria, palabras in categorias.items():

        for palabra in palabras:

            if palabra in texto:
                return categoria

    return "Medicina General"

# =========================================================
# EXTRACCIONES
# =========================================================

def extraer_dni(texto):

    dni = re.search(r'\b\d{8}\b', texto)

    return dni.group(0) if dni else None

def extraer_edad(texto):

    edad = re.search(
        r'(?:edad|años)\s*[:\-]?\s*(\d{1,3})',
        texto,
        re.IGNORECASE
    )

    return edad.group(1) if edad else None

def extraer_cmp(texto):

    cmp = re.search(
        r'(?:cmp)\s*[:\-]?\s*(\d+)',
        texto,
        re.IGNORECASE
    )

    return cmp.group(1) if cmp else None

def extraer_fechas(texto):

    fechas = re.findall(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        texto
    )

    return " | ".join(fechas) if fechas else None

def extraer_nombre(texto):

    patrones = [

        r'(?:PACIENTE|NOMBRE)[:\s]+([A-ZÁÉÍÓÚÑ ]+)'

    ]

    for patron in patrones:

        resultado = re.search(
            patron,
            texto,
            re.IGNORECASE
        )

        if resultado:

            return resultado.group(1).strip()

    return None

# =========================================================
# SENTIMIENTO
# =========================================================

def analizar_sentimiento(texto):

    try:

        polaridad = TextBlob(
            texto
        ).sentiment.polarity

        if polaridad > 0:
            return "Positivo"

        elif polaridad < 0:
            return "Negativo"

        else:
            return "Neutral"

    except:

        return "Neutral"

# =========================================================
# RESUMEN
# =========================================================

def generar_resumen(texto):

    palabras = texto.split()

    return " ".join(palabras[:25])

# =========================================================
# UPLOADER
# =========================================================

uploaded_files = st.file_uploader(

    "Subir Certificados Médicos",

    type=["jpg", "jpeg", "png"],

    accept_multiple_files=True

)

# =========================================================
# PREVIEW
# =========================================================

if uploaded_files:

    st.markdown("## Vista Previa")

    for file in uploaded_files[:8]:

        img = Image.open(file)

        st.markdown('<div class="preview-container">', unsafe_allow_html=True)

        st.image(
            img,
            width=350
        )

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PROCESAMIENTO
# =========================================================

if uploaded_files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados_finales = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file)

            pre = preprocesar_imagen(image)

            if mostrar_preprocesamiento:

                st.markdown(
                    f"## {uploaded_file.name}"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        image,
                        use_container_width=True
                    )

                with c2:

                    st.image(
                        pre['gray'],
                        caption="Escala de grises",
                        use_container_width=True
                    )

            resultado_ocr = reader.readtext(

                pre['gray'],

                detail=0,

                paragraph=False,

                batch_size=1

            )

            texto_ocr = "\n".join(resultado_ocr)

            resultado_nlp = pipeline_limpieza(texto_ocr)

            categoria = clasificar_documento(texto_ocr)

            sentimiento = analizar_sentimiento(texto_ocr)

            resumen = generar_resumen(
                resultado_nlp['texto_preprocesado']
            )

            resultados_finales.append({

                'archivo': uploaded_file.name,

                'nombre': extraer_nombre(texto_ocr),

                'dni': extraer_dni(texto_ocr),

                'edad': extraer_edad(texto_ocr),

                'cmp': extraer_cmp(texto_ocr),

                'fechas': extraer_fechas(texto_ocr),

                'categoria_tematica': categoria,

                'sentimiento': sentimiento,

                'resumen': resumen,

                'texto_ocr': texto_ocr,

                'texto_preprocesado':
                    resultado_nlp['texto_preprocesado'],

                'tokens_originales':
                    resultado_nlp['n_tokens_original'],

                'tokens_limpios':
                    resultado_nlp['n_tokens_limpios']

            })

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

            gc.collect()

        df_final = pd.DataFrame(resultados_finales)

        st.success("OCR + NLP FINALIZADO")

        texto_total = " ".join(
            df_final['texto_preprocesado']
        )

        palabras = texto_total.split()

        frecuencias = Counter(palabras)

        # =====================================================
        # METRICAS
        # =====================================================

        st.markdown("## Métricas")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card card1">
                <div class="metric-title">Documentos</div>
                <div class="metric-value">{len(df_final)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card card2">
                <div class="metric-title">Tokens</div>
                <div class="metric-value">{int(df_final['tokens_originales'].mean())}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card card3">
                <div class="metric-title">Limpios</div>
                <div class="metric-value">{int(df_final['tokens_limpios'].mean())}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card card4">
                <div class="metric-title">Categorías</div>
                <div class="metric-value">{df_final['categoria_tematica'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card card5">
                <div class="metric-title">Keywords</div>
                <div class="metric-value">{len(frecuencias)}</div>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

            "Resultados",
            "NLP",
            "WordCloud",
            "Clasificación",
            "Sentimientos",
            "Exportar"

        ])

        # =====================================================
        # RESULTADOS
        # =====================================================

        with tab1:

            st.markdown("## Resultados OCR + NLP")

            st.info(
                "Visualización resumida de entidades extraídas automáticamente mediante OCR y NLP."
            )

            columnas_visibles = [

                'archivo',
                'nombre',
                'dni',
                'edad',
                'categoria_tematica',
                'sentimiento',
                'cmp',
                'fechas'

            ]

            st.data_editor(

                df_final[columnas_visibles],

                use_container_width=True,

                hide_index=True,

                disabled=True

            )

            with st.expander("Ver información completa"):

                st.dataframe(
                    df_final,
                    use_container_width=True
                )

        # =====================================================
        # NLP
        # =====================================================

        with tab2:

            top20 = frecuencias.most_common(20)

            if len(top20) > 0:

                palabras_top, conteos = zip(*top20)

                fig, ax = plt.subplots(figsize=(12,6))

                ax.barh(
                    list(reversed(palabras_top)),
                    list(reversed(conteos))
                )

                ax.set_title("Top 20 Palabras")

                st.pyplot(fig)

        # =====================================================
        # WORDCLOUD
        # =====================================================

        with tab3:

            texto_final = " ".join(palabras)

            if texto_final.strip() != "":

                wordcloud = WordCloud(

                    width=1800,
                    height=900,

                    background_color='white',

                    max_words=40,

                    stopwords=STOPWORDS_COMPLETO,

                    collocations=False

                ).generate(texto_final)

                fig_wc, ax_wc = plt.subplots(figsize=(16,8))

                ax_wc.imshow(
                    wordcloud,
                    interpolation='bilinear'
                )

                ax_wc.axis("off")

                st.pyplot(fig_wc)

        # =====================================================
        # CLASIFICACION
        # =====================================================

        with tab4:

            conteo = df_final[
                'categoria_tematica'
            ].value_counts()

            fig2, ax2 = plt.subplots(figsize=(8,5))

            ax2.bar(
                conteo.index,
                conteo.values
            )

            ax2.set_title(
                "Clasificación Temática"
            )

            st.pyplot(fig2)

        # =====================================================
        # SENTIMIENTOS
        # =====================================================

        with tab5:

            conteo_sentimientos = df_final[
                'sentimiento'
            ].value_counts()

            fig3, ax3 = plt.subplots(figsize=(7,5))

            ax3.pie(
                conteo_sentimientos.values,
                labels=conteo_sentimientos.index,
                autopct='%1.1f%%'
            )

            ax3.set_title(
                "Análisis de Sentimiento"
            )

            st.pyplot(fig3)

        # =====================================================
        # EXPORTAR
        # =====================================================

        with tab6:

            csv = df_final.to_csv(
                index=False
            ).encode('utf-8-sig')

            st.download_button(
                "Descargar CSV",
                csv,
                "resultado_ocr.csv",
                "text/csv"
            )

            output = BytesIO()

            with pd.ExcelWriter(
                output,
                engine='openpyxl'
            ) as writer:

                df_final.to_excel(
                    writer,
                    index=False
                )

            st.download_button(
                "Descargar Excel",
                output.getvalue(),
                "resultado_ocr.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )