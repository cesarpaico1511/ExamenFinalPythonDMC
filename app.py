import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de estilo
st.set_page_config(
    page_title="Telco Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración uniforme de estilos gráficos para la organización
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 13})

# CLASE DE ARQUITECTURA DE DATOS (POO)
# =====================================================================
class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self._clean_and_transform()
        
    def _clean_and_transform(self):
        if 'TotalCharges' in self.df.columns:
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        if 'Churn' in self.df.columns:
            self.df['Churn'] = self.df['Churn'].astype(str).str.strip()

    def classify_variables(self):
        columns_to_check = [col for col in self.df.columns if col.lower() != 'customerid']
        num_vars = self.df[columns_to_check].select_dtypes(include=[np.number]).columns.tolist()
        cat_vars = self.df[columns_to_check].select_dtypes(include=['object', 'category']).columns.tolist()
        return num_vars, cat_vars

    def get_descriptive_stats(self, numeric_columns):
        return self.df[numeric_columns].describe().T

    def plot_histogram(self, column: str, kde: bool = True):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data=self.df, x=column, kde=kde, color="#1f77b4", ax=ax)
        ax.set_title(f"Distribución de: {column}")
        plt.tight_layout()
        return fig

    def plot_categorical_bar(self, column: str):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        order = self.df[column].value_counts().index
        sns.countplot(data=self.df, x=column, order=order, palette="Blues_r", ax=ax)
        ax.set_title(f"Frecuencias Absolutas: {column}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        return fig

    def plot_bivariate_box(self, num_col: str, cat_col: str = "Churn"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette="Set2", ax=ax)
        ax.set_title(f"Impacto de {num_col} vs {cat_col}")
        plt.tight_layout()
        return fig

    def plot_bivariate_stacked_bar(self, cat_col: str, target_col: str = "Churn"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        crosstab_data = pd.crosstab(self.df[cat_col], self.df[target_col], normalize='index') * 100
        crosstab_data.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=ax)
        ax.set_title(f"Proporción de Churn según {cat_col}")
        ax.set_ylabel("Porcentaje (%)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        return fig


# NAVEGACIÓN
# =====================================================================

st.image("images/innovacion.png", width=300)
st.sidebar.title("🧭 Panel de Control")
st.sidebar.markdown("---")

navigation_menu = st.sidebar.radio(
    "Seleccione un Módulo:",
    ["🏠 Home", "📂 Carga de Dataset", "📊 Análisis Exploratorio (EDA)"]
)

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# MÓDULO 1: HOME
if navigation_menu == "🏠 Home":
    st.title("🚀 Analítica de Retención de Clientes - Telecomunicaciones")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📋 Contexto de Negocio")
        st.write("Durante la coyuntura del COVID-19, la empresa incrementó su ratio de fuga de clientes en +0.5 puntos porcentuales, pasando de un promedio histórico del 2.0% a un 2.5%. Dado que el costo de adquirir un nuevo cliente es entre 6 y 7 veces mayor que retener uno existente, es de vital importancia contar con herramientas profesionales de Análisis Exploratorio de Datos (EDA). Esta aplicación web interactiva desarrollada sobre Streamlit permite identificar patrones críticos de comportamiento e indicios de abandono para sustentar la toma de decisiones estratégicas sin recurrir a modelos predictivos de caja negra.")
        st.markdown("### 🛠️ Tecnologías")
        st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit.")
    with col2:
        st.markdown("### 👤 Datos del Autor")
        st.success("**Autor:** Julio Cesar Paico Jaime\n\n**Especialización:** Python potenciado con IA\n\n**Año:** 2026")

# MÓDULO 2: CARGA
elif navigation_menu == "📂 Carga de Dataset":
    st.title("📂 Carga del Dataset")
    st.markdown("---")
    uploaded_file = st.file_uploader("Suba el archivo TelcoCustomerChurn.csv", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.raw_data = pd.read_csv(uploaded_file)
        st.success("✔️ Archivo cargado con éxito.")
        st.metric(label="Filas", value=f"{st.session_state.raw_data.shape[0]:,}")
        st.metric(label="Columnas", value=str(st.session_state.raw_data.shape[1]))
        st.dataframe(st.session_state.raw_data.head(5), use_container_width=True)
    else:
        st.warning("⚠️ Control de flujo: Suba el archivo CSV para desbloquear el análisis.")

# MÓDULO 3: EDA
elif navigation_menu == "📊 Análisis Exploratorio (EDA)":
    st.title("📊 Núcleo EDA")
    st.markdown("---")
    if st.session_state.raw_data is None:
        st.warning("⚠️ Control de flujo: Cargue los datos en el Módulo anterior.")
        st.stop()
        
    processor = DataProcessor(st.session_state.raw_data)
    num_vars, cat_vars = processor.classify_variables()
    
    t1, t2, t3, t4, t5 = st.tabs(["🔬 Estructura", "📈 Univariado", "👥 Bivariado", "⚙️ Dinámico", "🎯 Insights"])
    
    with t1:
        st.markdown("### Ítems 1 y 2: Información General y Clasificación")
        st.write(f"**Numéricas:** {num_vars}")
        st.write(f"**Categóricas:** {cat_vars}")
        st.markdown("### Ítem 3: Estadística Descriptiva")
        st.dataframe(processor.get_descriptive_stats(num_vars))
        st.markdown("### Ítem 4: Valores Faltantes")
        st.dataframe(processor.df.isnull().sum().to_frame(name="Nulos"))
        
    with t2:
        st.markdown("### Ítem 5: Distribución Numérica")
        sel_num = st.selectbox("Columna numérica:", num_vars)
        fig = processor.plot_histogram(sel_num)
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("### Ítem 6: Distribución Categórica")
        sel_cat = st.selectbox("Columna categórica:", cat_vars)
        fig2 = processor.plot_categorical_bar(sel_cat)
        st.pyplot(fig2)
        plt.close(fig2)
        
    with t3:
        st.markdown("### Ítem 7: Numérico vs Churn")
        b_num = st.selectbox("Métrica:", ["tenure", "MonthlyCharges"])
        fig3 = processor.plot_bivariate_box(b_num)
        st.pyplot(fig3)
        plt.close(fig3)
        
        st.markdown("### Ítem 8: Categórico vs Churn")
        b_cat = st.selectbox("Categoría:", ["Contract", "InternetService"])
        fig4 = processor.plot_bivariate_stacked_bar(b_cat)
        st.pyplot(fig4)
        plt.close(fig4)
        
    with t4:
        st.markdown("### Ítem 9: Filtros Interactivos")
        max_t = int(processor.df["tenure"].max())
        rango = st.slider("Meses de Permanencia:", 0, max_t, (0, max_t))
        filtro_df = processor.df[(processor.df["tenure"] >= rango[0]) & (processor.df["tenure"] <= rango[1])]
        st.write(f"Registros en rango: {filtro_df.shape[0]}")
        st.dataframe(filtro_df.head(5))
        
    with t5:
        st.markdown("### Ítem 10: Hallazgos Clave")
        fig5, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=processor.df, x="tenure", y="MonthlyCharges", hue="Churn", alpha=0.5, ax=ax)
        st.pyplot(fig5)
        plt.close(fig5)
        
        st.markdown("### 📌 Conclusiones Estratégicas")
        st.write("1. Los contratos 'Mes a Mes' concentran el mayor porcentaje de abandono.")
        st.write("2. Los clientes que cancelan muestran cargos mensuales superiores a los $70 USD.")
        st.write("3. La mayor densidad de fuga ocurre en los primeros 6 meses de antigüedad.")
        st.write("4. Clientes con Fibra Óptica exhiben tasas de churn proporcionalmente más altas.")
        st.write("5. Métodos de pago manuales (Electronic Check) presentan mayor tasa de abandono.")
