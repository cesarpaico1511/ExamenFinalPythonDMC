
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURACIÓN ESTRUCTURAL DE LA PÁGINA (Instrucción Primaria)
# =====================================================================
st.set_page_config(
    page_title="Telco Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración uniforme de estilos gráficos para la organización
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 13})

# =====================================================================
# ENCAPSULAMIENTO EN PROGRAMACIÓN ORIENTADA A OBJETOS (POO)
# =====================================================================
class DataAnalyzer:
    """
    Clase de nivel producción encargada de gestionar los procesos de limpieza,
    transformación tipológica, segmentación analítica y renderizado de gráficos.
    """
    def __init__(self, dataframe: pd.DataFrame):
        # Evitar mutabilidad no deseada del dataframe base mediante copia profunda
        self.df = dataframe.copy()
        self._preprocesar_datos_criticos()
        
    def _preprocesar_datos_criticos(self):
        """Manejo de inconsistencias de tipos de datos en variables financieras."""
        if 'TotalCharges' in self.df.columns:
            # Reemplazo seguro de espacios en blanco por estructuras NaN
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            
        if 'Churn' in self.df.columns:
            self.df['Churn'] = self.df['Churn'].astype(str).str.strip()

    def clasificar_variables_personalizada(self):
        """Función algorítmica para aislar tipos cualitativos y cuantitativos."""
        # Exclusión estricta del ID único del cliente para preservar consistencia estadística
        columnas_analiticas = [col for col in self.df.columns if col.lower() != 'customerid']
        
        num_vars = self.df[columnas_analiticas].select_dtypes(include=[np.number]).columns.tolist()
        cat_vars = self.df[columnas_analiticas].select_dtypes(include=['object', 'category']).columns.tolist()
        return num_vars, cat_vars

    def obtener_estadistica_descriptiva(self, columnas_numericas):
        """Genera matriz consolidada de medidas de tendencia central y dispersión."""
        return self.df[columnas_numericas].describe().T

    def graficar_histograma_univariado(self, columna: str, activar_kde: bool = True):
        """Renderiza distribución de frecuencias continuas."""
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.histplot(data=self.df, x=columna, kde=activar_kde, color="#1f77b4", ax=ax)
        ax.set_title(f"Distribución Univariada de: {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia Absoluta")
        plt.tight_layout()
        return fig

    def graficar_barras_categoricas(self, columna: str):
        """Renderiza frecuencias para atributos discretos ordenados de forma descendente."""
        fig, ax = plt.subplots(figsize=(6, 3.8))
        orden_frecuencia = self.df[columna].value_counts().index
        sns.countplot(data=self.df, x=columna, order=orden_frecuencia, palette="Blues_r", ax=ax)
        ax.set_title(f"Conteo de Categorías: {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Cantidad de Clientes")
        plt.xticks(rotation=15)
        plt.tight_layout()
        return fig

    def graficar_bivariado_caja(self, variable_num: str, variable_cat: str = "Churn"):
        """Muestra diagramas de caja y bigotes para evaluar contrastes de grupos."""
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(data=self.df, x=variable_cat, y=variable_num, palette="Set2", ax=ax)
        ax.set_title(f"Análisis Boxplot: {variable_num} vs {variable_cat}")
        ax.set_xlabel(f"Estatus de Fuga ({variable_cat})")
        ax.set_ylabel(variable_num)
        plt.tight_layout()
        return fig

    def graficar_bivariado_barras_apiladas(self, variable_cat: str, objetivo: str = "Churn"):
        """Genera tablas de contingencia relacionales normalizadas al 100%."""
        fig, ax = plt.subplots(figsize=(6, 3.8))
        tabla_contingencia = pd.crosstab(self.df[variable_cat], self.df[objetivo], normalize='index') * 100
        
        tabla_contingencia.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=ax)
        ax.set_title(f"Proporción de Churn Relativa por {variable_cat}")
        ax.set_xlabel(variable_cat)
        ax.set_ylabel("Porcentaje Proporcional (%)")
        ax.legend(title="Churn", loc="lower left")
        plt.xticks(rotation=15)
        plt.tight_layout()
        return fig

# =====================================================================
# SISTEMA GENERAL DE NAVEGACIÓN (CONTROL DE FLUJO)
# =====================================================================
st.sidebar.title("🧭 Navegación Estratégica")
st.sidebar.markdown("---")
modulo_seleccionado = st.sidebar.radio(
    "Seleccione Módulo de Trabajo:",
    ["🏠 Home / Presentación", "📂 Carga de Dataset", "📊 Análisis Exploratorio (EDA)"]
)

# Inicialización persistente del estado de sesión para el intercambio de información entre pestañas
if 'dataset_usuarios' not in st.session_state:
    st.session_state.dataset_usuarios = None

# -----------------------------------------------------------------
# MÓDULO 1: HOME (PRESENTACIÓN INSTITUCIONAL)
# -----------------------------------------------------------------
if modulo_seleccionado == "🏠 Home / Presentación":
    st.title("🚀 Plataforma de Analítica Corporativa de Retención de Clientes")
    st.subheader("Análisis de Diagnóstico sobre Factores de Deserción Contractual")
    st.markdown("---")
    
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown("### 📋 Justificación Comercial y Problemática")
        st.write(
            "Durante la ventana temporal correspondiente a la coyuntura global del COVID-19, la organización "
            "experimentó una elevación crítica en su métrica de fuga de clientes de **+0.5 puntos porcentuales**, "
            "escalando de un promedio controlado del **2.0% a un 2.5%**."
        )
        st.info(
            "💡 **Métrica Financiera Clave:** Los análisis de costo-beneficio evidencian que la tasa de adquisición de un "
            "nuevo usuario representa un costo financiero de entre **6 y 7 veces mayor** que los mecanismos operativos de "
            "retención de un cliente existente. Esta herramienta fue diseñada como un producto analítico real para "
            "sustentar la toma de decisiones estratégicas sin fines predictivos."
        )
        st.markdown("### 🛠️ Ecosistema Tecnológico de Soporte")
        st.markdown("- **Core:** Python  \n- **Procesamiento Estructural:** Pandas & NumPy  \n- **Librerías de Visualización:** Matplotlib & Seaborn  \n- **Despliegue de Interfaz:** Streamlit")
        
    with col_h2:
        st.markdown("### 👤 Ficha del Investigador")
        st.success("**Nombre del Alumno:** Alumno Experto  \n\n**Especialización:** Python potenciado con IA  \n\n**Año Académico:** 2026")
        st.markdown("### 📦 Atributos Generales")
        st.write("El conjunto de datos histórico evalúa características demográficas, tipos de servicios técnicos contratados, variables de facturación mensual y la condición final de deserción o permanencia.")

# -----------------------------------------------------------------
# MÓDULO 2: CARGA ASISTIDA DEL DATASET
# -----------------------------------------------------------------
elif modulo_seleccionado == "📂 Carga de Dataset":
    st.title("📂 Repositorio de Carga e Inspección Sanitaria")
    st.markdown("---")
    
    archivo_cargado = st.file_uploader("Por favor, seleccione el archivo fuente TelcoCustomerChurn.csv:", type=["csv"])
    
    if archivo_cargado is not None:
        try:
            df_fuente = pd.read_csv(archivo_cargado)
            st.session_state.dataset_usuarios = df_fuente
            st.success("✔️ Archivo cargado correctamente e indexado en el buffer del sistema.")
            
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.metric(label="Volumen Total de Clientes Analizados (Filas)", value=f"{df_fuente.shape[0]:,}")
            with c_m2:
                st.metric(label="Dimensiones del Atributo de Datos (Columnas)", value=str(df_fuente.shape[1]))
                
            st.markdown("### 🗂️ Vista Previa de Registros Estructurados (Primeras 5 Filas)")
            st.dataframe(df_fuente.head(5), use_container_width=True)
        except Exception as error:
            st.error(f"Fallo crítico al procesar el archivo estructurado: {error}")
    else:
        st.warning("⚠️ Control de Flujo Activado: El sistema se encuentra en espera de la carga del archivo CSV. Suba el dataset para desbloquear las funciones analíticas.")

# -----------------------------------------------------------------
# MÓDULO 3: NÚCLEO CORE - ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# -----------------------------------------------------------------
elif modulo_seleccionado == "📊 Análisis Exploratorio (EDA)":
    st.title("📊 Núcleo Integrado de Análisis Exploratorio de Datos (EDA)")
    st.markdown("---")
    
    # REGLA CRÍTICA DE CONTROL DE FLUJO: Validación de persistencia de datos
    if st.session_state.dataset_usuarios is None:
        st.warning("⚠️ Acceso Denegado: No se ha detectado información estructurada en memoria. Diríjase al menú '📂 Carga de Dataset' para habilitar las funciones analíticas.")
        st.stop()
        
    # Instanciación de la clase bajo el paradigma POO
    analizador = DataAnalyzer(st.session_state.dataset_usuarios)
    columnas_num, columnas_cat = analizador.clasificar_variables_personalizada()
    
    # Arquitectura limpia de visualización organizada en pestañas temáticas
    tab_est, tab_uni, tab_biv, tab_din, tab_ins = st.tabs([
        "🔬 Diagnóstico Estructural", "📈 Análisis Univariado", "👥 Cruce Bivariado", "⚙️ Simulación Dinámica", "🎯 Hallazgos Clave"
    ])
    
    # --- PESTAÑA 1: DIAGNÓSTICO ESTRUCTURAL (Ítems 1 al 4) ---
    with tab_est:
        st.markdown("### Ítem 1: Mapeo de Tipos de Datos y Muestra")
        resumen_estructura = pd.DataFrame({
            "Estructura Atributo": analizador.df.columns,
            "Tipo de Almacenamiento": [str(t) for t in analizador.df.dtypes],
            "Registros Habilitados Non-Null": analizador.df.notnull().sum().values
        })
        st.dataframe(resumen_estructura, use_container_width=True)
        
        st.markdown("### Ítem 2: Clasificación Algorítmica de Variables")
        st.write(f"**Atributos Cuantitativos (`{len(columnas_num)}`):** {', '.join(columnas_num)}")
        st.write(f"**Atributos Cualitativos (`{len(columnas_cat)}`):** {', '.join(columnas_cat[:5])}... y {len(columnas_cat)-5} adicionales.")
        
        st.markdown("### Ítem 3: Consolidado Estadístico Descriptivo")
        st.dataframe(analizador.obtener_estadistica_descriptiva(columnas_num), use_container_width=True)
        st.write("**Interpretación:** La variable `tenure` muestra una media de 32.3 meses de permanencia con alta variabilidad. La mediana se sitúa en 29 meses, denotando un sesgo provocado por los extremos transaccionales.")
        
        st.markdown("### Ítem 4: Análisis de Valores Faltantes y Nulos")
        conteos_nulos = analizador.df.isnull().sum()
        porcentajes_nulos = (analizador.df.isnull().sum() / len(analizador.df)) * 100
        tabla_sanidad = pd.DataFrame({"Cantidad de Vacíos": conteos_nulos, "Porcentaje Relativo (%)": porcentajes_nulos})
        st.dataframe(tabla_sanidad, use_container_width=True)
        st.write("**Discusión Técnica:** Los registros vacíos aislados en `TotalCharges` corresponden a cuentas con `tenure = 0`. Esto indica que son usuarios con contratos recién firmados que no han pasado por el primer ciclo de facturación mensual.")

    # --- PESTAÑA 2: ANALISIS UNIVARIADO (Ítems 5 y 6) ---
    with tab_uni:
        c_u1, c_u2 = st.columns(2)
        with c_u1:
            st.markdown("### Ítem 5: Distribución de Características Numéricas")
            num_seleccionada = st.selectbox("Seleccione la columna numérica:", columnas_num, key="uni_n")
            activar_densidad = st.checkbox("Superponer Curva KDE", value=True)
            
            fig_h = analizador.graficar_histograma_univariado(num_seleccionada, activar_densidad)
            st.pyplot(fig_h)
            plt.close(fig_h)
        with c_u2:
            st.markdown("### Ítem 6: Frecuencias de Características Categóricas")
            cat_seleccionada = st.selectbox("Seleccione la columna categórica:", columnas_cat, key="uni_c")
            
            fig_b = analizador.graficar_barras_categoricas(cat_seleccionada)
            st.pyplot(fig_b)
            plt.close(fig_b)

    # --- PESTAÑA 3: ANALISIS BIVARIADO (Ítems 7 y 8) ---
    with tab_biv:
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            st.markdown("### Ítem 7: Análisis Bivariado (Numérico vs Churn)")
            biv_num = st.selectbox("Seleccione Métrica Financiera:", ["tenure", "MonthlyCharges", "TotalCharges"], key="biv_n")
            
            fig_box = analizador.graficar_bivariado_caja(biv_num)
            st.pyplot(fig_box)
            plt.close(fig_box)
            st.write("Se observa que los usuarios en estatus de fuga activa registran una permanencia notablemente inferior y cargos mensuales superiores en comparación con el segmento retenido.")
        with c_b2:
            st.markdown("### Ítem 8: Análisis Bivariado (Categórico vs Churn)")
            biv_cat = st.selectbox("Seleccione Atributo Comercial:", ["Contract", "InternetService", "PaymentMethod"], key="biv_c")
            
            fig_apilada = analizador.graficar_bivariado_barras_apiladas(biv_cat)
            st.pyplot(fig_apilada)
            plt.close(fig_apilada)
            st.write("Los gráficos apilados confirman un patrón crítico de deserción en los contratos Month-to-month y en servicios provistos por tecnología de Fibra Óptica.")

    # --- PESTAÑA 4: EXPLORACIÓN DINÁMICA DE PARÁMETROS (Ítem 9) ---
    with tab_din:
        st.markdown("### Ítem 9: Análisis Dinámico Basado en Parámetros Seleccionados")
        st.write("Modifique los controles paramétricos interconectados para aislar el comportamiento de subsegmentos del mercado en tiempo real.")
        
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            limite_tenure = st.slider("Ventana de Antigüedad Evaluada (Meses):", 0, int(analizador.df["tenure"].max()), (0, 36))
        with c_f2:
            opciones_contrato = st.multiselect("Esquemas Contractuales Habilitados:", options=analizador.df["Contract"].unique().tolist(), default=analizador.df["Contract"].unique().tolist())
            
        # Ejecución matemática del filtro interactivo
        df_segmentado = analizador.df[
            (analizador.df["tenure"] >= limite_tenure[0]) & 
            (analizador.df["tenure"] <= limite_tenure[1]) & 
            (analizador.df["Contract"].isin(opciones_contrato))
        ]
        
        st.metric(label="Muestra Coincidente", value=f"{df_segmentado.shape[0]} registros")
        if not df_segmentado.empty and "Churn" in df_segmentado.columns:
            tasa_fuga_segmento = (df_segmentado["Churn"].value_counts(normalize=True).get("Yes", 0.0)) * 100
            st.metric(label="Tasa Focalizada de Churn en el Subsegmento", value=f"{tasa_fuga_segmento:.2f}%")
            st.dataframe(df_segmentado.head(5), use_container_width=True)

    # --- PESTAÑA 5: INSIGHTS ESTRATÉGICOS (Ítem 10) ---
    with tab_ins:
        st.markdown("### Ítem 10: Matriz Resumen de Hallazgos Clave")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            fig_scat, ax_scat = plt.subplots(figsize=(6, 4.2))
            sns.scatterplot(data=analizador.df, x="tenure", y="MonthlyCharges", hue="Churn", palette={"Yes": "#d62728", "No": "#2ca02c"}, alpha=0.4, ax=ax_scat)
            ax_scat.set_title("Estructura Espacial del Abandono de Clientes")
            st.pyplot(fig_scat)
            plt.close(fig_scat)
        with col_res2:
            st.markdown("#### 🎯 Indicios Identificados en la Población")
            st.write(
                "El análisis de dispersión cruzado revela una alta densidad de puntos de deserción (marcas rojas) "
                "localizados en la zona de **baja permanencia (0 a 12 meses)** vinculados a **altas tarifas de facturación "
                "mensual (superiores a $70 USD)**.\n\n"
                "Esto demuestra de forma empírica el foco de insatisfacción comercial inmediata post-COVID-19."
            )
            
        st.markdown("---")
        st.markdown("### 📌 Conclusiones Finales Orientadas a la Toma de Decisiones")
        st.markdown(
            "1. **Reestructuración Contractual de Corto Plazo:** Los contratos 'Mes a Mes' concentran de forma masiva el abandono. Se debe implementar una campaña comercial que financie ofertas de descuento para incentivar la conversión obligatoria a esquemas anuales estables.\n"
            "2. **Establecimiento de Alertas de Elasticidad de Precio:** Se evidencia un umbral de deserción acelerado al superar los $70 USD mensuales. Es necesario implantar alertas automáticas en el CRM para ofrecer empaquetamientos competitivos antes de que el cliente finalice su relación.\n"
            "3. **Mecanismo de Retención Temprana:** Al ser los primeros 6 meses el ciclo de vida con mayor densidad de Churn, se aconseja programar campañas de soporte especializado y encuestas de satisfacción inmediatas durante este periodo crítico.\n"
            "4. **Auditoría Técnica en Canales de Fibra Óptica:** A pesar de ser un producto tecnológico premium, el servicio de Internet por Fibra Óptica registra tasas de Churn anormalmente elevadas frente a DSL,
