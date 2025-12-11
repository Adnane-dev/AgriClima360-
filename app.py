# =============================================================
# AGRICLIMA360 - Application Streamlit avec données NOAA API
# Visualisations climatiques interactives AVEC ANIMATIONS
# et VISUALISATIONS MASSIVES (Dask/Datashader/hvPlot/Panel)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import time
import io
import base64
import tempfile
from fpdf import FPDF

# Import des librairies pour visualisations massives
try:
    import dask.dataframe as dd
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.colors import viridis
    import hvplot.pandas
    import hvplot.dask
    import holoviews as hv
    import panel as pn
    from holoviews.operation.datashader import datashade, dynspread
    from holoviews import streams
    hv.extension('bokeh')
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    st.warning("⚠️ Pour les visualisations massives, installez: `pip install dask datashader holoviews hvplot panel bokeh`")

# Configuration de la page
st.set_page_config(
    page_title="AgriClima360 - Dashboard Climatique Avancé",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# 1. CONFIGURATION API NOAA
# =============================================================

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/"

# ⚠️ IMPORTANT: Obtenez votre token gratuit sur https://www.ncdc.noaa.gov/cdo-web/token
NOAA_TOKEN = st.secrets.get("NOAA_TOKEN", "YOUR_TOKEN_HERE")

@st.cache_data(ttl=3600)
def get_noaa_data(endpoint, params=None, token=NOAA_TOKEN):
    headers = {"token": token}
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None

@st.cache_data(ttl=3600)
def get_climate_data(dataset_id="GHCND", start_date="2020-01-01", end_date="2023-12-31", 
                     location_id=None, datatypes=None, limit=1000):
    params = {
        "datasetid": dataset_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": limit,
    }
    
    if location_id:
        params["locationid"] = location_id
    
    if datatypes:
        params["datatypeid"] = ",".join(datatypes)
    
    data = get_noaa_data("data", params)
    
    if data and "results" in data:
        df = pd.DataFrame(data["results"])
        return df
    
    return pd.DataFrame()

# =============================================================
# 2. FONCTIONS DE TRAITEMENT AVANCÉES
# =============================================================

def process_climate_data(df):
    """Traite et enrichit les données climatiques avec plus de variables."""
    if df.empty:
        return generate_enhanced_sample_data()
    
    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Conversion des températures (de dixièmes de degrés Celsius)
    if 'value' in df.columns:
        # Les températures NOAA sont en dixièmes de degrés
        temp_types = ['TMAX', 'TMIN', 'TAVG']
        df.loc[df['datatype'].isin(temp_types), 'value'] = df.loc[df['datatype'].isin(temp_types), 'value'] / 10
        
        # Les précipitations sont en dixièmes de mm
        df.loc[df['datatype'] == 'PRCP', 'value'] = df.loc[df['datatype'] == 'PRCP', 'value'] / 10
    
    # Pivoter pour avoir les différents types de données en colonnes
    df_pivot = df.pivot_table(
        index=['date', 'year', 'month', 'day', 'day_of_year', 'station'],
        columns='datatype',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Renommer les colonnes
    column_mapping = {
        'TMAX': 'tmax',
        'TMIN': 'tmin',
        'TAVG': 'tavg',
        'PRCP': 'prcp',
        'SNOW': 'snow',
        'SNWD': 'snow_depth',
        'AWND': 'wind_avg',
        'WSF2': 'wind_fastest'
    }
    # Renommer uniquement les colonnes existantes
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=existing_columns)
    
    # Calculer tavg si manquant
    if 'tavg' not in df_pivot.columns and 'tmax' in df_pivot.columns and 'tmin' in df_pivot.columns:
        df_pivot['tavg'] = (df_pivot['tmax'] + df_pivot['tmin']) / 2
    
    # Ajouter des données simulées pour les visualisations avancées
    df_pivot['humidity'] = np.random.uniform(30, 90, len(df_pivot))
    df_pivot['wind_speed'] = np.random.uniform(0, 20, len(df_pivot))
    df_pivot['solar_radiation'] = np.random.uniform(100, 800, len(df_pivot))
    df_pivot['continent'] = np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania'], len(df_pivot))
    df_pivot['lat'] = 40.0 + np.random.uniform(-5, 5, len(df_pivot))
    df_pivot['lon'] = -100.0 + np.random.uniform(-10, 10, len(df_pivot))
    df_pivot['elevation'] = np.random.uniform(0, 3000, len(df_pivot))
    
    return df_pivot

def generate_enhanced_sample_data(num_points=500000):
    """Génère des données de démonstration enrichies à grande échelle."""
    st.warning("Configurez votre token NOAA pour des données réelles.")
    
    years = list(range(2000, 2026))
    stations = [f'ST{i:03d}' for i in range(1, 201)]  # 200 stations
    continents = ['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania']
    
    # Utiliser numpy pour génération rapide
    data = {
        'date': [],
        'year': [],
        'month': [],
        'day': [],
        'station': [],
        'tavg': [],
        'tmax': [],
        'tmin': [],
        'prcp': [],
        'humidity': [],
        'wind_speed': [],
        'solar_radiation': [],
        'continent': [],
        'lat': [],
        'lon': [],
        'elevation': []
    }
    
    # Générer des données pour chaque année
    for year in years:
        n_samples = num_points // len(years)
        
        # Générer des dates aléatoires dans l'année
        dates = pd.to_datetime([f'{year}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}' 
                               for _ in range(n_samples)])
        
        data['date'].extend(dates)
        data['year'].extend([year] * n_samples)
        data['month'].extend(dates.month)
        data['day'].extend(dates.day)
        data['station'].extend(np.random.choice(stations, n_samples))
        
        # Températures avec tendance au réchauffement
        warming_trend = 0.03 * (year - 2020)
        base_temp = 15 + warming_trend
        seasonal_variation = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
        
        tavg = base_temp + seasonal_variation + np.random.normal(0, 2, n_samples)
        data['tavg'].extend(tavg)
        data['tmax'].extend(tavg + 5 + np.random.normal(0, 1, n_samples))
        data['tmin'].extend(tavg - 5 + np.random.normal(0, 1, n_samples))
        
        # Autres variables
        data['prcp'].extend(np.random.exponential(5, n_samples))
        data['humidity'].extend(np.random.uniform(30, 90, n_samples))
        data['wind_speed'].extend(np.random.uniform(0, 20, n_samples))
        data['solar_radiation'].extend(np.random.uniform(100, 800, n_samples))
        data['continent'].extend(np.random.choice(continents, n_samples))
        data['lat'].extend(np.random.uniform(-90, 90, n_samples))
        data['lon'].extend(np.random.uniform(-180, 180, n_samples))
        data['elevation'].extend(np.random.uniform(0, 3000, n_samples))
    
    return pd.DataFrame(data)

# =============================================================
# 3. FONCTIONS POUR VISUALISATIONS MASSIVES
# =============================================================

def create_datashader_scatter(df_dask, x_col='tavg', y_col='prcp', color_col='year', 
                             width=800, height=600, cmap='viridis'):
    """Crée une visualisation Datashader pour des millions de points."""
    if not DASH_AVAILABLE:
        return None
    
    try:
        # Convertir en dataframe Dask si nécessaire
        if not isinstance(df_dask, dd.DataFrame):
            df_dask = dd.from_pandas(df_dask, npartitions=10)
        
        # Préparer les données
        x = df_dask[x_col].compute()
        y = df_dask[y_col].compute()
        
        if color_col in df_dask.columns:
            color = df_dask[color_col].compute()
        else:
            color = None
        
        # Créer le canvas Datashader
        canvas = ds.Canvas(plot_width=width, plot_height=height)
        
        # Agréger les points
        if color is not None:
            agg = canvas.points(pd.DataFrame({x_col: x, y_col: y}), x_col, y_col, ds.mean(color_col))
        else:
            agg = canvas.points(pd.DataFrame({x_col: x, y_col: y}), x_col, y_col)
        
        # Appliquer la colormap
        img = tf.shade(agg, cmap=plt.cm.get_cmap(cmap))
        img = tf.set_background(img, 'white')
        
        # Convertir en image PIL pour affichage
        from PIL import Image
        img_pil = Image.fromarray(np.array(img.to_pil()))
        
        return img_pil
        
    except Exception as e:
        st.error(f"Erreur Datashader: {e}")
        return None

def create_hvplot_large_data(df, x='tavg', y='prcp', color='year', 
                           datashade=True, width=800, height=500):
    """Crée une visualisation hvPlot avec Datashader pour grands datasets."""
    if not DASH_AVAILABLE:
        return None
    
    try:
        # Créer le plot avec hvPlot
        plot = df.hvplot.scatter(
            x=x, 
            y=y, 
            c=color,
            cmap='viridis',
            colorbar=True,
            title=f'{y} vs {x} ({len(df):,} points)',
            width=width,
            height=height,
            alpha=0.3 if not datashade else 1.0,
            datashade=datashade,
            dynspread=True if datashade else False
        )
        
        return plot
        
    except Exception as e:
        st.error(f"Erreur hvPlot: {e}")
        return None

def create_density_heatmap_datashader(df_dask, x_col='lon', y_col='lat', z_col='tavg',
                                     width=800, height=600, cmap='hot'):
    """Crée une carte de densité avec Datashader."""
    if not DASH_AVAILABLE:
        return None
    
    try:
        if not isinstance(df_dask, dd.DataFrame):
            df_dask = dd.from_pandas(df_dask, npartitions=10)
        
        # Calculer les agrégations
        canvas = ds.Canvas(plot_width=width, plot_height=height)
        
        if z_col:
            # Heatmap avec valeur moyenne
            agg = canvas.points(df_dask.compute(), x_col, y_col, ds.mean(z_col))
            img = tf.shade(agg, cmap=plt.cm.get_cmap(cmap))
        else:
            # Simple comptage de points
            agg = canvas.points(df_dask.compute(), x_col, y_col)
            img = tf.shade(agg, cmap=plt.cm.get_cmap(cmap))
        
        img = tf.set_background(img, 'black')
        
        # Convertir pour affichage
        from PIL import Image
        img_pil = Image.fromarray(np.array(img.to_pil()))
        
        return img_pil
        
    except Exception as e:
        st.error(f"Erreur densité Datashader: {e}")
        return None

def create_interactive_large_map(df, lat_col='lat', lon_col='lon', value_col='tavg',
                                aggregation='mean', tiles='CartoDark', width=800, height=600):
    """Crée une carte interactive pour grands datasets."""
    if not DASH_AVAILABLE:
        return None
    
    try:
        # Utiliser hvPlot avec Datashader pour la carte
        map_plot = df.hvplot.points(
            x=lon_col,
            y=lat_col,
            c=value_col,
            cmap='viridis',
            geo=True,
            tiles=tiles,
            alpha=0.5,
            frame_width=width,
            frame_height=height,
            title=f'Carte des {value_col} ({len(df):,} points)',
            datashade=True,
            aggregator=aggregation
        )
        
        return map_plot
        
    except Exception as e:
        st.error(f"Erreur carte interactive: {e}")
        return None

def create_time_series_aggregation(df_dask, time_col='date', value_col='tavg',
                                 freq='M', aggregation='mean', width=800, height=400):
    """Crée une série temporelle agrégée pour grands datasets."""
    if not DASH_AVAILABLE:
        return None
    
    try:
        if not isinstance(df_dask, dd.DataFrame):
            df_dask = dd.from_pandas(df_dask, npartitions=10)
        
        # Agrégation temporelle avec Dask
        df_dask[time_col] = dd.to_datetime(df_dask[time_col])
        df_dask = df_dask.set_index(time_col)
        
        if aggregation == 'mean':
            aggregated = df_dask[value_col].resample(freq).mean().compute()
        elif aggregation == 'sum':
            aggregated = df_dask[value_col].resample(freq).sum().compute()
        elif aggregation == 'max':
            aggregated = df_dask[value_col].resample(freq).max().compute()
        else:  # min
            aggregated = df_dask[value_col].resample(freq).min().compute()
        
        # Créer le plot avec hvPlot
        plot = aggregated.hvplot(
            width=width,
            height=height,
            title=f'{value_col} ({aggregation}) par {freq}',
            line_width=2,
            grid=True,
            ylabel=value_col,
            xlabel='Date'
        )
        
        return plot
        
    except Exception as e:
        st.error(f"Erreur série temporelle: {e}")
        return None

def create_parallel_coordinates_large(df_sample, cols=None, alpha=0.1, width=1000, height=500):
    """Crée un diagramme de coordonnées parallèles pour grands datasets."""
    if not DASH_AVAILABLE or len(df_sample) > 10000:
        # Pour très grands datasets, on échantillonne
        df_sample = df_sample.sample(min(10000, len(df_sample)))
    
    try:
        if cols is None:
            cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
        
        cols = [c for c in cols if c in df_sample.columns]
        
        # Créer le plot avec hvPlot
        plot = df_sample.hvplot.parallel_coordinates(
            dimensions=cols,
            label='Variables climatiques',
            width=width,
            height=height,
            alpha=alpha
        )
        
        return plot
        
    except Exception as e:
        st.error(f"Erreur coordonnées parallèles: {e}")
        return None

# =============================================================
# 4. INTERFACE STREAMLIT AVANCÉE AVEC VISUALISATIONS MASSIVES
# =============================================================

def main():
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("🌾 AgriClima360")
        st.markdown("### Dashboard Climatique Avancé")
        st.markdown("*Avec visualisations massives*")
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Sélection de la source de données
        data_source = st.radio(
            "Source de données:",
            ["API NOAA (Réelles)", "Démonstration Grande Échelle"]
        )
        
        if data_source == "Démonstration Grande Échelle":
            data_size = st.select_slider(
                "Taille du dataset:",
                options=["100K", "500K", "1M", "2M"],
                value="500K"
            )
            size_map = {"100K": 100000, "500K": 500000, "1M": 1000000, "2M": 2000000}
            sample_size = size_map[data_size]
        
        if data_source == "API NOAA (Réelles)":
            st.info("ℹ️ Token NOAA requis")
            
            with st.expander("📡 Paramètres API NOAA"):
                dataset = st.selectbox(
                    "Dataset:",
                    ["GHCND", "GSOM", "GSOY"],
                    help="GHCND = Données quotidiennes, GSOM = Mensuelles, GSOY = Annuelles"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Date début:",
                        value=datetime(2020, 1, 1),
                        min_value=datetime(1900, 1, 1)
                    )
                
                with col2:
                    end_date = st.date_input(
                        "Date fin:",
                        value=datetime(2023, 12, 31),
                        max_value=datetime.now()
                    )
                
                location = st.text_input(
                    "Localisation (optionnel):",
                    placeholder="ex: FIPS:US"
                )
                
                datatypes = st.multiselect(
                    "Types de données:",
                    ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSF2"],
                    default=["TMAX", "TMIN", "PRCP", "AWND"]
                )
                
                limit = st.slider("Nombre de résultats:", 100, 1000000, 10000)
        
        st.markdown("---")
        
        # Navigation avec nouvelle option pour visualisations massives
        st.header("📊 Navigation")
        page = st.radio(
            "Sections:",
            ["🏠 Vue d'ensemble", "📈 Analyses Animées", "🌐 Visualisations 3D", 
             "🗺️ Carte Animée", "🚀 Visualisations Massives", "🔬 Avancé", "🎯 Radar & Parallèles"]
        )
        
        st.markdown("---")
        
        # Filtres
        st.header("🎛️ Filtres")
        year_filter = st.empty()
        continent_filter = st.empty()
        
        st.markdown("---")
        
        # Boutons d'export
        st.header("💾 Export")
        export_format = st.selectbox("Format d'export:", ["CSV", "JSON", "Excel", "PDF Rapport"])
        
        st.markdown("---")
        
        # Information sur les librairies
        if DASH_AVAILABLE:
            st.success("✅ Visualisations massives activées")
        else:
            st.warning("⚠️ Visualisations massives non disponibles")
            st.info("Installez: `pip install dask datashader holoviews hvplot panel bokeh`")
    
    # Chargement des données
    with st.spinner("⏳ Chargement des données enrichies..."):
        if data_source == "API NOAA (Réelles)":
            if NOAA_TOKEN == "YOUR_TOKEN_HERE":
                df = generate_enhanced_sample_data(100000)
            else:
                raw_data = get_climate_data(
                    dataset_id=dataset,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location_id=location if location else None,
                    datatypes=datatypes if datatypes else None,
                    limit=limit
                )
                df = process_climate_data(raw_data)
        else:
            df = generate_enhanced_sample_data(sample_size)
    
    # Vérification des données
    if df.empty:
        st.error("❌ Aucune donnée disponible. Vérifiez vos paramètres.")
        return
    
    # Afficher les statistiques du dataset
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Statistiques Dataset:**")
    st.sidebar.markdown(f"• Points de données: **{len(df):,}**")
    st.sidebar.markdown(f"• Période: **{df['year'].min()} - {df['year'].max()}**")
    st.sidebar.markdown(f"• Stations: **{df['station'].nunique()}**")
    st.sidebar.markdown(f"• Colonnes: **{len(df.columns)}**")
    
    # Calcul des KPIs
    kpis = compute_kpis(df)
    
    # Filtres dans la sidebar
    with st.sidebar:
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            if len(years) > 0:
                selected_years = year_filter.slider(
                    "Période:",
                    int(min(years)),
                    int(max(years)),
                    (int(min(years)), int(max(years)))
                )
                df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
        
        if 'continent' in df.columns:
            continents = ['Tous'] + sorted(df['continent'].unique().tolist())
            selected_continent = continent_filter.selectbox(
                "Continent:",
                continents
            )
            if selected_continent != 'Tous':
                df = df[df['continent'] == selected_continent]
    
    # =============================================================
    # NOUVELLE PAGE : VISUALISATIONS MASSIVES
    # =============================================================
    
    if page == "🚀 Visualisations Massives":
        st.title("🚀 Visualisations Massives")
        st.markdown(f"### Analyse de {len(df):,} points de données")
        
        if not DASH_AVAILABLE:
            st.error("""
            ❌ Les librairies pour visualisations massives ne sont pas installées.
            
            **Installation requise:**
            ```bash
            pip install dask datashader holoviews hvplot panel bokeh
            ```
            
            Redémarrez l'application après l'installation.
            """)
            return
        
        # Informations sur les performances
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Points de données", f"{len(df):,}")
        with col_info2:
            st.metric("Taille mémoire", f"{(df.memory_usage().sum() / 1024 / 1024):.1f} MB")
        with col_info3:
            st.metric("Technologie", "Dask/Datashader")
        
        st.markdown("---")
        
        # Sélection du type de visualisation
        viz_type = st.selectbox(
            "Type de visualisation:",
            ["Nuage de points massif", "Carte de densité", "Série temporelle agrégée", 
             "Coordonnées parallèles", "Heatmap géographique", "Comparaison multivariée"]
        )
        
        # Options spécifiques selon le type
        if viz_type == "Nuage de points massif":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Variable X:", df.select_dtypes(include=[np.number]).columns.tolist(), index=0)
            with col2:
                y_var = st.selectbox("Variable Y:", df.select_dtypes(include=[np.number]).columns.tolist(), index=1)
            with col3:
                color_var = st.selectbox("Variable couleur:", ['year', 'month', 'continent', 'tavg'])
            
            use_datashader = st.checkbox("Utiliser Datashader (recommandé pour > 50K points)", value=len(df) > 50000)
            
            if st.button("🔄 Générer la visualisation"):
                with st.spinner("Génération de la visualisation massive..."):
                    if use_datashader:
                        # Utiliser Datashader pour visualisation massive
                        img = create_datashader_scatter(
                            df, 
                            x_col=x_var, 
                            y_col=y_var, 
                            color_col=color_var,
                            width=1000, 
                            height=600
                        )
                        
                        if img:
                            st.image(img, caption=f"Datashader: {y_var} vs {x_var} ({len(df):,} points)")
                            st.info("ℹ️ Datashader agrège les points pour une visualisation optimale des grands datasets")
                    else:
                        # Utiliser hvPlot avec datashading
                        plot = create_hvplot_large_data(
                            df, 
                            x=x_var, 
                            y=y_var, 
                            color=color_var,
                            datashade=True,
                            width=1000, 
                            height=600
                        )
                        
                        if plot:
                            # Convertir le plot HoloViews en HTML pour l'affichage
                            import holoviews as hv
                            hv.save(plot, 'temp_plot.html')
                            with open('temp_plot.html', 'r') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=650)
        
        elif viz_type == "Carte de densité":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Variable X (géographique):", ['lon', 'lat', 'tavg', 'prcp'], index=0)
            with col2:
                y_var = st.selectbox("Variable Y (géographique):", ['lat', 'lon', 'humidity', 'elevation'], index=1)
            with col3:
                z_var = st.selectbox("Variable valeur:", ['tavg', 'prcp', 'humidity', 'wind_speed'], index=0)
            
            colormap = st.selectbox("Colormap:", ['viridis', 'plasma', 'hot', 'coolwarm', 'rainbow'])
            
            if st.button("🔄 Générer la carte de densité"):
                with st.spinner("Création de la carte de densité..."):
                    img = create_density_heatmap_datashader(
                        df, 
                        x_col=x_var, 
                        y_col=y_var, 
                        z_col=z_var,
                        width=1000, 
                        height=600,
                        cmap=colormap
                    )
                    
                    if img:
                        st.image(img, caption=f"Carte de densité: {z_var} par {x_var}/{y_var}")
                        st.info("ℹ️ Chaque pixel représente la valeur moyenne dans cette zone")
        
        elif viz_type == "Série temporelle agrégée":
            col1, col2, col3 = st.columns(3)
            with col1:
                value_var = st.selectbox("Variable à analyser:", ['tavg', 'tmax', 'tmin', 'prcp'], index=0)
            with col2:
                freq = st.selectbox("Fréquence d'agrégation:", ['D', 'W', 'M', 'Q', 'Y'], index=2)
                freq_names = {'D': 'Jour', 'W': 'Semaine', 'M': 'Mois', 'Q': 'Trimestre', 'Y': 'Année'}
            with col3:
                aggregation = st.selectbox("Type d'agrégation:", ['mean', 'sum', 'max', 'min'], index=0)
            
            if st.button("🔄 Générer la série temporelle"):
                with st.spinner("Aggrégation des données temporelles..."):
                    # Convertir en Dask DataFrame pour le traitement
                    df_dask = dd.from_pandas(df, npartitions=10)
                    
                    plot = create_time_series_aggregation(
                        df_dask,
                        time_col='date',
                        value_col=value_var,
                        freq=freq,
                        aggregation=aggregation,
                        width=1000,
                        height=500
                    )
                    
                    if plot:
                        # Afficher les statistiques d'agrégation
                        st.success(f"✅ Données agrégées par {freq_names[freq]} ({aggregation})")
                        
                        # Convertir et afficher le plot
                        import holoviews as hv
                        hv.save(plot, 'temp_timeseries.html')
                        with open('temp_timeseries.html', 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=550)
        
        elif viz_type == "Coordonnées parallèles":
            st.info("ℹ️ Cette visualisation échantillonne automatiquement les données pour > 10K points")
            
            # Sélection des variables
            available_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed', 'solar_radiation', 'elevation']
            selected_cols = st.multiselect(
                "Variables à inclure:",
                available_cols,
                default=['tavg', 'tmax', 'tmin', 'prcp', 'humidity']
            )
            
            alpha = st.slider("Transparence des lignes:", 0.01, 0.5, 0.1, 0.01)
            
            if st.button("🔄 Générer les coordonnées parallèles"):
                with st.spinner("Construction du diagramme..."):
                    # Échantillonner pour les très grands datasets
                    if len(df) > 10000:
                        df_sample = df.sample(10000)
                        st.warning(f"⚠️ Échantillonnage à 10K points pour la lisibilité (sur {len(df):,})")
                    else:
                        df_sample = df
                    
                    plot = create_parallel_coordinates_large(
                        df_sample,
                        cols=selected_cols,
                        alpha=alpha,
                        width=1000,
                        height=500
                    )
                    
                    if plot:
                        import holoviews as hv
                        hv.save(plot, 'temp_parallel.html')
                        with open('temp_parallel.html', 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=550)
        
        elif viz_type == "Heatmap géographique":
            st.markdown("### 🌍 Visualisation Géographique Massive")
            
            col1, col2 = st.columns(2)
            with col1:
                value_var = st.selectbox("Variable à cartographier:", ['tavg', 'prcp', 'humidity', 'wind_speed'], index=0)
                aggregation = st.selectbox("Méthode d'agrégation:", ['mean', 'max', 'min', 'count'], index=0)
            
            with col2:
                tiles = st.selectbox("Fond de carte:", ['CartoLight', 'CartoDark', 'OSM', 'EsriImagery'], index=0)
                point_size = st.slider("Taille des points:", 1, 20, 5)
            
            if st.button("🗺️ Générer la carte interactive"):
                with st.spinner("Construction de la carte géographique..."):
                    # Échantillonner pour les très grands datasets
                    if len(df) > 100000:
                        df_sample = df.sample(100000)
                        st.warning(f"⚠️ Échantillonnage à 100K points pour la performance (sur {len(df):,})")
                    else:
                        df_sample = df
                    
                    map_plot = create_interactive_large_map(
                        df_sample,
                        lat_col='lat',
                        lon_col='lon',
                        value_col=value_var,
                        aggregation=aggregation,
                        tiles=tiles,
                        width=1000,
                        height=600
                    )
                    
                    if map_plot:
                        import holoviews as hv
                        hv.save(map_plot, 'temp_map.html')
                        with open('temp_map.html', 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=650)
        
        elif viz_type == "Comparaison multivariée":
            st.markdown("### 📊 Analyse Multivariée Massive")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variable X:", df.select_dtypes(include=[np.number]).columns.tolist(), index=0)
                y_var = st.selectbox("Variable Y:", df.select_dtypes(include=[np.number]).columns.tolist(), index=1)
            
            with col2:
                color_var = st.selectbox("Variable pour la couleur:", ['continent', 'year', 'month'], index=0)
                size_var = st.selectbox("Variable pour la taille:", ['prcp', 'wind_speed', 'solar_radiation', 'None'], index=0)
            
            use_hexbin = st.checkbox("Utiliser Hexbin pour l'agrégation", value=True)
            
            if st.button("🔍 Générer l'analyse multivariée"):
                with st.spinner("Analyse des corrélations..."):
                    if use_hexbin:
                        # Créer un hexbin plot avec hvPlot
                        plot = df.hvplot.hexbin(
                            x=x_var,
                            y=y_var,
                            C=color_var if color_var != 'None' else None,
                            width=1000,
                            height=600,
                            title=f"Hexbin: {y_var} vs {x_var}",
                            gridsize=30,
                            cmap='viridis'
                        )
                    else:
                        # Scatter plot avec datashading
                        plot = create_hvplot_large_data(
                            df,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            datashade=True,
                            width=1000,
                            height=600
                        )
                    
                    if plot:
                        import holoviews as hv
                        hv.save(plot, 'temp_multivar.html')
                        with open('temp_multivar.html', 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=650)
                        
                        # Ajouter des statistiques de corrélation
                        if x_var in df.columns and y_var in df.columns:
                            corr = df[x_var].corr(df[y_var])
                            st.metric("Corrélation", f"{corr:.3f}")
        
        # Section d'information sur les technologies
        st.markdown("---")
        with st.expander("ℹ️ Technologies utilisées pour les visualisations massives"):
            st.markdown("""
            ### 🚀 Technologies de Visualisation Massive
            
            **Dask**: 
            - Traitement parallèle des données
            - Gestion de datasets plus grands que la mémoire RAM
            - API similaire à Pandas
            
            **Datashader**:
            - Rendue de millions de points en temps réel
            - Agrégation intelligente pour éviter le surpeuplement
            - Préservation des tendances statistiques
            
            **hvPlot / HoloViews**:
            - Interface de visualisation déclarative
            - Intégration avec Datashader
            - Graphiques interactifs haute performance
            
            **Panel**:
            - Création de dashboards interactifs
            - Intégration avec Streamlit
            - Widgets interactifs pour l'exploration
            
            **Performance**:
            - Jusqu'à 10 millions de points visibles
            - Temps de rendu < 2 secondes
            - Utilisation mémoire optimisée
            """)
    
   

# =============================================================
# FONCTIONS UTILITAIRES MANQUANTES (à ajouter)
# =============================================================

def compute_kpis(df):
    """Calcule les indicateurs clés avancés."""
    kpis = {}
    
    if not df.empty:
        kpis["temp_moy"] = df["tavg"].mean() if "tavg" in df.columns else 0
        kpis["pluie_totale"] = df["prcp"].sum() if "prcp" in df.columns else 0
        kpis["nb_annees"] = df["year"].nunique()
        kpis["temp_max"] = df["tmax"].max() if "tmax" in df.columns else 0
        kpis["temp_min"] = df["tmin"].min() if "tmin" in df.columns else 0
        kpis["humidite_moy"] = df["humidity"].mean() if "humidity" in df.columns else 65
        kpis["solar_avg"] = df["solar_radiation"].mean() if "solar_radiation" in df.columns else 0
        kpis["wind_avg"] = df["wind_speed"].mean() if "wind_speed" in df.columns else 0
        
        # Calcul de la tendance de température
        if "tavg" in df.columns and df['year'].nunique() > 1:
            yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
            if len(yearly_avg) > 1:
                coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                kpis["temp_trend"] = coeffs[0] * 100  # °C par siècle
            else:
                kpis["temp_trend"] = 0
        else:
            kpis["temp_trend"] = 0
            
        # Calcul de la variabilité
        if "tavg" in df.columns and df['year'].nunique() > 1:
            kpis["variability"] = df.groupby('year')['tavg'].std().mean()
        else:
            kpis["variability"] = 0
            
        # Calcul des canicules
        if "tmax" in df.columns and len(df) > 0:
            kpis["heatwaves"] = (df['tmax'] > 30).sum() / len(df) * 100
        else:
            kpis["heatwaves"] = 0
            
        # Calcul du risque de sécheresse
        if "prcp" in df.columns and len(df) > 0:
            kpis["drought_risk"] = (df['prcp'] < 5).sum() / len(df) * 100
        else:
            kpis["drought_risk"] = 0
            
        # Nombre de continents
        if "continent" in df.columns:
            kpis["continents"] = df["continent"].nunique()
        else:
            kpis["continents"] = 1
    
    return kpis

# ... [Ajoutez ici toutes vos autres fonctions existantes comme create_temperature_evolution, etc.] ...


# =============================================================
# 3. FONCTIONS DE VISUALISATION AVANCÉES (ANIMATIONS)
# =============================================================

def create_temperature_evolution(df):
    """Crée le graphique d'évolution des températures avec animation."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    yearly_data = df.groupby('year').agg({
        'tavg': 'mean',
        'tmax': 'max',
        'tmin': 'min'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tmax'],
        name='Température Max',
        mode='lines+markers',
        line=dict(color='red', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Max</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tavg'],
        name='Température Moyenne',
        mode='lines+markers',
        line=dict(color='orange', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Moy</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tmin'],
        name='Température Min',
        mode='lines+markers',
        line=dict(color='blue', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Min</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title='📈 Évolution des Températures (Interactive)',
        xaxis_title='Année',
        yaxis_title='Température (°C)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_precipitation_chart(df):
    """Crée le graphique des précipitations avec interactivité."""
    if df.empty or 'prcp' not in df.columns:
        return go.Figure()
    
    monthly_prcp = df.groupby(['year', 'month'])['prcp'].sum().reset_index()
    
    fig = px.bar(
        monthly_prcp,
        x='month',
        y='prcp',
        color='year',
        title='💧 Précipitations Mensuelles (Animées)',
        labels={'month': 'Mois', 'prcp': 'Précipitations (mm)', 'year': 'Année'},
        height=500,
        animation_frame='year',
        range_y=[0, monthly_prcp['prcp'].max() * 1.1] if not monthly_prcp.empty else [0, 100]
    )
    
    return fig

def create_animated_temperature_map(df):
    """Crée une carte animée des températures."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    yearly_avg = df.groupby(['year', 'continent']).agg({
        'tavg': 'mean',
        'tmax': 'max',
        'tmin': 'min',
        'prcp': 'sum',
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    fig = px.scatter_geo(yearly_avg,
                        lat='lat',
                        lon='lon',
                        color='tavg',
                        size='prcp',
                        animation_frame='year',
                        hover_name='continent',
                        hover_data=['tavg', 'tmax', 'tmin', 'prcp'],
                        color_continuous_scale=px.colors.sequential.Viridis,
                        projection='natural earth',
                        title='🌡️ Évolution Mondiale des Températures (Animée)',
                        height=600)
    
    fig.update_layout(geo=dict(showland=True, landcolor="lightgray"))
    
    return fig

def create_3d_scatter_plot(df):
    """Crée un graphique 3D interactif."""
    if df.empty:
        return go.Figure()
    
    sample_df = df.sample(min(1000, len(df)))
    
    fig = px.scatter_3d(sample_df,
                       x='tavg',
                       y='prcp',
                       z='humidity',
                       color='continent',
                       size='solar_radiation' if 'solar_radiation' in df.columns else 'wind_speed',
                       hover_name='station',
                       title='🌐 Visualisation 3D Interactive des Variables Climatiques',
                       height=600)
    
    fig.update_layout(scene=dict(
        xaxis_title='Température Moyenne (°C)',
        yaxis_title='Précipitations (mm)',
        zaxis_title='Humidité (%)'
    ))
    
    return fig

def create_interactive_heatmap(df):
    """Crée une heatmap interactive avec zoom."""
    if df.empty or 'tavg' not in df.columns:
        return go.Figure()
    
    pivot_data = df.pivot_table(index='month', columns='year', values='tavg', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        colorscale='Viridis',
        colorbar=dict(title="Température (°C)"),
        hoverongaps=False,
        hovertemplate='Année: %{x}<br>Mois: %{y}<br>Température: %{z:.1f}°C<extra></extra>',
        zsmooth='best'
    ))
    
    fig.update_layout(
        title='📅 Heatmap Interactive des Températures',
        xaxis_title="Année",
        yaxis_title="Mois",
        height=500,
        dragmode='zoom',
        hovermode='closest'
    )
    
    return fig

def create_radar_chart(df, year=None):
    """Crée un graphique radar pour une année spécifique."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    if year is None:
        year = df['year'].max()
    
    year_data = df[df['year'] == year]
    
    if len(year_data) == 0:
        return go.Figure()
    
    # Vérifier que toutes les colonnes nécessaires existent
    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
    missing_cols = [col for col in required_cols if col not in year_data.columns]
    
    if missing_cols:
        # Créer des colonnes manquantes avec des valeurs par défaut
        for col in missing_cols:
            if col == 'prcp':
                year_data[col] = 0
            elif col in ['tavg', 'tmax', 'tmin']:
                year_data[col] = 20
            elif col == 'humidity':
                year_data[col] = 50
            elif col == 'wind_speed':
                year_data[col] = 5
    
    avg_data = year_data[required_cols].mean()
    
    # Normaliser les données pour le radar
    max_vals = df[required_cols].max()
    min_vals = df[required_cols].min()
    
    normalized_data = (avg_data - min_vals) / (max_vals - min_vals)
    
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            normalized_data['tavg'],
            normalized_data['tmax'],
            normalized_data['tmin'],
            normalized_data['prcp'] / 100,  # Réduire l'échelle des précipitations
            normalized_data['humidity'] / 100,
            normalized_data['wind_speed'] / 20
        ],
        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
        fill='toself',
        name=f'Année {year}',
        line_color='blue',
        opacity=0.8
    ))
    
    # Ajouter des données de référence (moyenne historique)
    ref_data = df[required_cols].mean()
    normalized_ref = (ref_data - min_vals) / (max_vals - min_vals)
    
    fig.add_trace(go.Scatterpolar(
        r=[
            normalized_ref['tavg'],
            normalized_ref['tmax'],
            normalized_ref['tmin'],
            normalized_ref['prcp'] / 100,
            normalized_ref['humidity'] / 100,
            normalized_ref['wind_speed'] / 20
        ],
        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
        fill='toself',
        name='Moyenne historique',
        line_color='gray',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                direction="clockwise"
            )
        ),
        showlegend=True,
        title=f'📊 Profil Climatique {year} (Graphique Radar)',
        height=500
    )
    
    return fig

def create_parallel_coordinates(df, selected_years=None):
    """Crée un diagramme de coordonnées parallèles."""
    if df.empty:
        return go.Figure()
    
    if selected_years:
        plot_df = df[df['year'].isin(selected_years)]
    else:
        plot_df = df.sample(min(500, len(df)))
    
    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed', 'year']
    available_cols = [col for col in required_cols if col in plot_df.columns]
    
    if 'year' not in available_cols:
        available_cols.append('year')
    
    fig = px.parallel_coordinates(plot_df,
                                 dimensions=available_cols[:-1],  # Exclure 'year' des dimensions
                                 color='year',
                                 labels={'tavg': 'Temp Moy', 'tmax': 'Temp Max',
                                        'tmin': 'Temp Min', 'prcp': 'Précip',
                                        'humidity': 'Humidité', 'wind_speed': 'Vent'},
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 title='📈 Coordonnées Parallèles des Variables Climatiques',
                                 height=500)
    
    return fig

def create_stream_graph(df):
    """Crée un graphique stream (courbes empilées)."""
    if df.empty or 'year' not in df.columns or 'month' not in df.columns:
        return go.Figure()
    
    monthly_data = df.groupby(['year', 'month']).agg({
        'tavg': 'mean',
        'prcp': 'sum'
    }).reset_index()
    
    # Pivoter pour le format stream
    stream_data = monthly_data.pivot(index='month', columns='year', values='tavg')
    
    fig = go.Figure()
    
    for year in stream_data.columns:
        fig.add_trace(go.Scatter(
            x=stream_data.index,
            y=stream_data[year],
            mode='lines',
            stackgroup='one',
            name=str(year),
            hoverinfo='x+y+name',
            line=dict(width=0.5),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title='🌊 Évolution des Températures (Graphique Stream)',
        xaxis_title='Mois',
        yaxis_title='Température Moyenne (°C)',
        showlegend=True,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_matrix_interactive(df):
    """Crée une matrice de corrélation interactive."""
    numeric_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed', 'solar_radiation']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return go.Figure()
    
    corr = df[available_cols].corr()
    
    # Créer une heatmap avec annotations
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12, "color": "black"},
        colorbar=dict(title="Corrélation"),
        hoverongaps=False,
        hovertemplate='<b>Variable X</b>: %{x}<br><b>Variable Y</b>: %{y}<br><b>Corrélation</b>: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🔗 Matrice de Corrélation Interactive',
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig

# =============================================================
# NOUVELLE FONCTION : GÉNÉRATION DE PDF
# =============================================================

def generate_pdf_report(df, kpis, temp_fig, heatmap_fig, map_fig, radar_fig):
    """
    Génère un rapport PDF avec les données et graphiques climatiques.
    
    Args:
        df: DataFrame contenant les données climatiques
        kpis: Dictionnaire des indicateurs clés
        temp_fig: Figure Plotly de l'évolution des températures
        heatmap_fig: Figure Plotly de la heatmap
        map_fig: Figure Plotly de la carte animée
        radar_fig: Figure Plotly du graphique radar
    
    Returns:
        bytes: Contenu du PDF à télécharger
    """
    
    # Créer un PDF
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1 : Page de titre et résumé
    pdf.add_page()
    
    # Titre
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(34, 139, 34)  # Vert forêt
    pdf.cell(0, 20, '🌍 Rapport AgriClima360', ln=1, align='C')
    
    # Sous-titre
    pdf.set_font('Helvetica', 'I', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Dashboard Climatique Avancé', ln=1, align='C')
    
    # Date
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}', ln=1, align='C')
    
    pdf.ln(10)
    
    # Ligne de séparation
    pdf.set_draw_color(34, 139, 34)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(15)
    
    # Résumé exécutif
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, '📊 Résumé Exécutif', ln=1)
    
    pdf.set_font('Helvetica', '', 12)
    resume_text = f"""
    Ce rapport présente une analyse climatique complète basée sur les données collectées.
    Période analysée: {df['year'].min()} - {df['year'].max()}
    Nombre de points de données: {len(df):,}
    Nombre d'années: {kpis.get('nb_annees', 0)}
    Régions couvertes: {kpis.get('continents', 1)} continent(s)
    """
    pdf.multi_cell(0, 8, resume_text)
    pdf.ln(10)
    
    # Indicateurs Clés
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '📈 Indicateurs Clés de Performance', ln=1)
    
    # Tableau des KPIs
    pdf.set_font('Helvetica', 'B', 12)
    col_widths = [70, 40, 40, 40]
    headers = ['Indicateur', 'Valeur', 'Unité', 'Tendance']
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C', fill=True)
    pdf.ln()
    
    # Données des KPIs
    pdf.set_font('Helvetica', '', 11)
    kpi_rows = [
        ['Température Moyenne', f"{kpis.get('temp_moy', 0):.1f}", '°C', f"{kpis.get('temp_trend', 0):+.2f}°C/siècle"],
        ['Précipitations Totales', f"{kpis.get('pluie_totale', 0):,.0f}", 'mm', f"{kpis.get('nb_annees', 0)} années"],
        ['Température Maximum', f"{kpis.get('temp_max', 0):.1f}", '°C', '-'],
        ['Température Minimum', f"{kpis.get('temp_min', 0):.1f}", '°C', '-'],
        ['Humidité Moyenne', f"{kpis.get('humidite_moy', 0):.1f}", '%', '-'],
        ['Radiation Solaire', f"{kpis.get('solar_avg', 0):.0f}", 'W/m²', '-'],
        ['Vitesse du Vent', f"{kpis.get('wind_avg', 0):.1f}", 'm/s', '-'],
        ['Jours de Canicule', f"{kpis.get('heatwaves', 0):.1f}", '%', '-'],
        ['Risque de Sécheresse', f"{kpis.get('drought_risk', 0):.1f}", '%', '-']
    ]
    
    for row in kpi_rows:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 8, cell, border=1, align='C')
        pdf.ln()
    
    pdf.ln(10)
    
    # Page 2 : Graphiques
    pdf.add_page()
    
    # Graphique 1 : Évolution des températures
    if temp_fig and temp_fig.data:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '📈 Évolution des Températures', ln=1)
        
        # Sauvegarder le graphique en image temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            temp_fig.write_image(tmpfile.name, width=800, height=400)
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
            pdf.ln(100)
            os.unlink(tmpfile.name)
    
    pdf.ln(10)
    
    # Graphique 2 : Heatmap
    if heatmap_fig and heatmap_fig.data:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '📅 Heatmap des Températures', ln=1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            heatmap_fig.write_image(tmpfile.name, width=800, height=400)
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
            pdf.ln(100)
            os.unlink(tmpfile.name)
    
    # Page 3 : Suite des graphiques
    pdf.add_page()
    
    # Graphique 3 : Carte
    if map_fig and map_fig.data:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '🗺️ Carte des Températures Mondiales', ln=1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            # Pour la carte, on peut réduire la taille pour mieux tenir dans le PDF
            map_fig.update_layout(height=400)
            map_fig.write_image(tmpfile.name, width=800, height=400)
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
            pdf.ln(100)
            os.unlink(tmpfile.name)
    
    pdf.ln(10)
    
    # Graphique 4 : Radar
    if radar_fig and radar_fig.data:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '📊 Graphique Radar', ln=1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            radar_fig.write_image(tmpfile.name, width=600, height=400)
            pdf.image(tmpfile.name, x=30, y=pdf.get_y(), w=150)
            pdf.ln(100)
            os.unlink(tmpfile.name)
    
    # Page 4 : Statistiques détaillées
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '📋 Statistiques Détailées', ln=1)
    
    if not df.empty:
        # Statistiques pour les variables numériques principales
        numeric_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed', 'solar_radiation']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            pdf.set_font('Helvetica', 'B', 12)
            headers = ['Variable', 'Moyenne', 'Médiane', 'Min', 'Max', 'Écart-type']
            col_widths = [40, 30, 30, 25, 25, 35]
            
            # En-tête
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, align='C', fill=True)
            pdf.ln()
            
            # Données
            pdf.set_font('Helvetica', '', 10)
            for col in available_cols[:8]:  # Limiter à 8 variables
                if col == 'tavg':
                    label = 'Temp Moy (°C)'
                elif col == 'tmax':
                    label = 'Temp Max (°C)'
                elif col == 'tmin':
                    label = 'Temp Min (°C)'
                elif col == 'prcp':
                    label = 'Précip (mm)'
                elif col == 'humidity':
                    label = 'Humidité (%)'
                elif col == 'wind_speed':
                    label = 'Vent (m/s)'
                elif col == 'solar_radiation':
                    label = 'Ray. Sol. (W/m²)'
                else:
                    label = col[:15]
                
                values = [
                    label,
                    f"{df[col].mean():.2f}",
                    f"{df[col].median():.2f}",
                    f"{df[col].min():.2f}",
                    f"{df[col].max():.2f}",
                    f"{df[col].std():.2f}"
                ]
                
                for i, value in enumerate(values):
                    pdf.cell(col_widths[i], 8, value, border=1, align='C')
                pdf.ln()
    
    pdf.ln(10)
    
    # Recommandations
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '🎯 Recommandations', ln=1)
    
    pdf.set_font('Helvetica', '', 11)
    recommendations = """
    1. Surveiller régulièrement les indicateurs de température et précipitations
    2. Adapter les pratiques agricoles aux tendances climatiques observées
    3. Mettre en place des systèmes d'alerte précoce pour les événements extrêmes
    4. Diversifier les cultures pour réduire la vulnérabilité climatique
    5. Intégrer les données climatiques dans la planification agricole
    """
    pdf.multi_cell(0, 8, recommendations)
    
    # Pied de page
    pdf.set_y(-30)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, 'Rapport généré automatiquement par AgriClima360', ln=1, align='C')
    pdf.cell(0, 8, 'Pour des analyses plus détaillées, consultez le dashboard interactif', ln=1, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# =============================================================
# 4. INTERFACE STREAMLIT AVANCÉE
# =============================================================

def main():
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("🌾 AgriClima360")
        st.markdown("### Dashboard Climatique Avancé")
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Sélection de la source de données
        data_source = st.radio(
            "Source de données:",
            ["API NOAA (Réelles)", "Démonstration"]
        )
        
        if data_source == "API NOAA (Réelles)":
            st.info("ℹ️ Token NOAA requis")
            
            # Configuration des paramètres NOAA
            with st.expander("📡 Paramètres API NOAA"):
                dataset = st.selectbox(
                    "Dataset:",
                    ["GHCND", "GSOM", "GSOY"],
                    help="GHCND = Données quotidiennes, GSOM = Mensuelles, GSOY = Annuelles"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Date début:",
                        value=datetime(2020, 1, 1),
                        min_value=datetime(1900, 1, 1)
                    )
                
                with col2:
                    end_date = st.date_input(
                        "Date fin:",
                        value=datetime(2023, 12, 31),
                        max_value=datetime.now()
                    )
                
                location = st.text_input(
                    "Localisation (optionnel):",
                    placeholder="ex: FIPS:US"
                )
                
                datatypes = st.multiselect(
                    "Types de données:",
                    ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSF2"],
                    default=["TMAX", "TMIN", "PRCP", "AWND"]
                )
                
                limit = st.slider("Nombre de résultats:", 100, 10000, 1000)
        
        st.markdown("---")
        
        # Navigation
        st.header("📊 Navigation")
        page = st.radio(
            "Sections:",
            ["🏠 Vue d'ensemble", "📈 Analyses Animées", "🌐 Visualisations 3D", 
             "🗺️ Carte Animée", "🔬 Avancé", "🎯 Radar & Parallèles"]
        )
        
        st.markdown("---")
        
        # Filtres
        st.header("🎛️ Filtres")
        
        # Filtre par années (sera appliqué après chargement)
        year_filter = st.empty()
        
        # Filtre par continent
        continent_filter = st.empty()
        
        st.markdown("---")
        
        # Contrôles d'animation
        st.header("🎬 Contrôles d'Animation")
        animation_speed = st.slider("Vitesse d'animation:", 100, 2000, 500, 100)
        auto_play = st.checkbox("Lecture automatique", value=True)
        
        # Boutons d'export
        st.header("💾 Export")
        export_format = st.selectbox("Format d'export:", ["CSV", "JSON", "Excel"])
    
    # Chargement des données
    with st.spinner("⏳ Chargement des données enrichies..."):
        if data_source == "API NOAA (Réelles)":
            if NOAA_TOKEN == "YOUR_TOKEN_HERE":
                st.error("❌ Token NOAA non configuré. Créez un fichier `.streamlit/secrets.toml` avec:\n```toml\nNOAA_TOKEN = 'votre_token'\n```")
                df = generate_enhanced_sample_data()
            else:
                raw_data = get_climate_data(
                    dataset_id=dataset,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location_id=location if location else None,
                    datatypes=datatypes if datatypes else None,
                    limit=limit
                )
                df = process_climate_data(raw_data)
        else:
            df = generate_enhanced_sample_data()
    
    # Vérification des données
    if df.empty:
        st.error("❌ Aucune donnée disponible. Vérifiez vos paramètres.")
        return
    
    # Calcul des KPIs
    kpis = compute_kpis(df)
    
    # Filtres dans la sidebar (maintenant qu'on a les données)
    with st.sidebar:
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            if len(years) > 0:
                selected_years = year_filter.slider(
                    "Période:",
                    int(min(years)),
                    int(max(years)),
                    (int(min(years)), int(max(years)))
                )
                df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
        
        if 'continent' in df.columns:
            continents = ['Tous'] + sorted(df['continent'].unique().tolist())
            selected_continent = continent_filter.selectbox(
                "Continent:",
                continents
            )
            if selected_continent != 'Tous':
                df = df[df['continent'] == selected_continent]
    
    # Vérifier à nouveau si le dataframe n'est pas vide après filtrage
    if df.empty:
        st.error("❌ Aucune donnée disponible après filtrage. Ajustez vos critères.")
        return
    
    # =============================================================
    # PAGES AVEC ANIMATIONS
    # =============================================================
    
    if page == "🏠 Vue d'ensemble":
        st.title("🌍 AgriClima360 - Dashboard Climatique Avancé")
        st.markdown("### Visualisations interactives avec animations")
        
        # KPIs en ligne
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🌡️ Température Moy.",
                f"{kpis.get('temp_moy', 0):.1f}°C",
                f"{kpis.get('temp_trend', 0):+.2f}°C/siècle"
            )
        
        with col2:
            st.metric(
                "💧 Précipitations",
                f"{kpis.get('pluie_totale', 0):,.0f} mm",
                f"{kpis.get('nb_annees', 0)} années"
            )
        
        with col3:
            st.metric(
                "⚠️ Canicules",
                f"{kpis.get('heatwaves', 0):.1f}%",
                f"Max: {kpis.get('temp_max', 0):.1f}°C"
            )
        
        with col4:
            st.metric(
                "🌞 Radiation Solaire",
                f"{kpis.get('solar_avg', 0):.0f} W/m²",
                f"Vent: {kpis.get('wind_avg', 0):.1f} m/s"
            )
        
        with col5:
            if "continents" in kpis:
                st.metric("🌐 Continents", f"{kpis.get('continents', 1)}", "Données globales")
        
        st.markdown("---")
        
        # Graphiques principaux avec animations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Évolution Temporelle (Animée)")
            fig_temp = create_temperature_evolution(df)
            st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        
        with col2:
            st.markdown("#### 💧 Précipitations (Animées)")
            fig_prcp = create_precipitation_chart(df)
            st.plotly_chart(fig_prcp, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        
        # Heatmap interactive
        st.markdown("#### 📅 Heatmap Interactive")
        fig_heatmap = create_interactive_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        st.plotly_chart(fig_temp, use_container_width=True)
        # Instructions pour les animations
        with st.expander("ℹ️ Comment utiliser les animations"):
            st.markdown("""
            ### Contrôles d'animation :
            1. **Boutons Play/Pause** : En haut à gauche des graphiques animés
            2. **Zoom** : Maintenez le clic et déplacez pour zoomer
            3. **Déplacement** : Cliquez sur l'icône de déplacement (main) en haut à droite
            4. **Réinitialiser** : Double-cliquez sur le graphique
            5. **Capture d'écran** : Cliquez sur l'appareil photo en haut à droite
            
            ### Fonctionnalités interactives :
            - **Survol** : Passez la souris pour voir les valeurs détaillées
            - **Sélection** : Cliquez et faites glisser pour sélectionner une zone
            - **Zoom** : Utilisez la molette de la souris ou pincez sur mobile
            """)
    
    elif page == "📈 Analyses Animées":
        st.title("📊 Analyses avec Animations")
        
        tab1, tab2, tab3 = st.tabs(["🌡️ Températures", "💧 Précipitations", "🔗 Corrélations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Évolution Temporelle Animée")
                fig_temp = create_temperature_evolution(df)
                
            
            with col2:
                st.markdown("#### Heatmap Interactive")
                fig_heatmap = create_interactive_heatmap(df)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Graphique stream
            st.markdown("#### Graphique Stream (Courbes Empilées)")
            fig_stream = create_stream_graph(df)
            st.plotly_chart(fig_stream, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Précipitations Animées")
                fig_prcp = create_precipitation_chart(df)
                st.plotly_chart(fig_prcp, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution des Précipitations")
                fig_box = px.box(df, x='year', y='prcp', title="📦 Distribution Annuelle des Précipitations")
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab3:
            st.markdown("#### Matrice de Corrélation Interactive")
            fig_corr = create_correlation_matrix_interactive(df)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Statistiques descriptives avec style
            st.markdown("#### 📊 Statistiques Descriptives Avancées")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_df = df[numeric_cols].describe().T
                stats_df['CV'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
                stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
                
                st.dataframe(stats_df, use_container_width=True)
    
    elif page == "🌐 Visualisations 3D":
        st.title("🌐 Visualisations 3D Interactives")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Visualisation 3D des Variables Climatiques")
            fig_3d = create_3d_scatter_plot(df)
            
            # Ajouter des contrôles 3D
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Température (°C)',
                    yaxis_title='Précipitations (mm)',
                    zaxis_title='Humidité (%)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            st.markdown("#### 🎮 Contrôles 3D")
            
            st.markdown("**Instructions :**")
            st.markdown("""
            - **Rotation** : Cliquez et faites glisser
            - **Zoom** : Molette de la souris
            - **Déplacement** : Maintenez Maj + glisser
            - **Réinitialiser** : Double-clic
            """)
            
            # Options de visualisation 3D
            st.markdown("**Options d'affichage :**")
            
            size_options = ['solar_radiation', 'wind_speed', 'prcp', 'tavg']
            size_options = [opt for opt in size_options if opt in df.columns]
            if size_options:
                size_var = st.selectbox(
                    "Taille des points par:",
                    size_options
                )
            else:
                size_var = None
            
            color_options = ['continent', 'year', 'month', 'tavg']
            color_options = [opt for opt in color_options if opt in df.columns]
            if color_options:
                color_var = st.selectbox(
                    "Couleur par:",
                    color_options
                )
            else:
                color_var = None
            
            z_options = ['humidity', 'prcp', 'wind_speed', 'solar_radiation']
            z_options = [opt for opt in z_options if opt in df.columns]
            if z_options:
                z_var = st.selectbox(
                    "Axe Z:",
                    z_options
                )
            else:
                z_var = 'humidity'
            
            if st.button("🔄 Mettre à jour la vue 3D") and size_var and color_var:
                sample_data = df.sample(min(1000, len(df)))
                fig_custom = px.scatter_3d(sample_data,
                                          x='tavg',
                                          y='prcp',
                                          z=z_var,
                                          color=color_var,
                                          size=size_var,
                                          title='🌐 Vue 3D Personnalisée',
                                          height=500)
                
                st.plotly_chart(fig_custom, use_container_width=True)
    
    elif page == "🗺️ Carte Animée":
        st.title("🗺️ Carte Climatique Animée")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 🌍 Carte Mondiale Interactive")
            fig_map = create_animated_temperature_map(df)
            
            st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            st.markdown("#### 🎛️ Contrôles de la Carte")
            
            map_type = st.selectbox(
                "Type de visualisation:",
                ['Températures', 'Précipitations', 'Risques', 'Zones']
            )
            
            point_size = st.slider("Taille des points:", 3, 20, 8)
            map_opacity = st.slider("Opacité:", 0.1, 1.0, 0.8, 0.1)
            
            projection = st.selectbox(
                "Projection:",
                ['natural earth', 'equirectangular', 'orthographic', 'mercator']
            )
            
            if st.button("🗺️ Actualiser la carte"):
                # Recréer la carte avec les nouveaux paramètres
                yearly_avg = df.groupby(['year', 'continent']).agg({
                    'tavg': 'mean',
                    'prcp': 'sum',
                    'lat': 'mean',
                    'lon': 'mean'
                }).reset_index()
                
                color_col = 'tavg' if map_type == 'Températures' else 'prcp'
                title = f'🌍 {map_type} - Animation Mondiale'
                
                if not yearly_avg.empty:
                    fig_custom_map = px.scatter_geo(yearly_avg,
                                                   lat='lat',
                                                   lon='lon',
                                                   color=color_col,
                                                   size='prcp',
                                                   animation_frame='year',
                                                   color_continuous_scale='Viridis',
                                                   projection=projection,
                                                   title=title,
                                                   height=500,
                                                   opacity=map_opacity)
                    
                    fig_custom_map.update_traces(marker=dict(size=point_size))
                    st.plotly_chart(fig_custom_map, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Statistiques Géographiques :**")
            st.metric("📍 Points de données", f"{len(df):,}")
            if 'lat' in df.columns:
                st.metric("🌐 Étendue Lat.", f"{df['lat'].max() - df['lat'].min():.1f}°")
            if 'lon' in df.columns:
                st.metric("🌐 Étendue Lon.", f"{df['lon'].max() - df['lon'].min():.1f}°")
    
    elif page == "🎯 Radar & Parallèles":
        st.title("🎯 Visualisations Avancées")
        
        tab1, tab2, tab3 = st.tabs(["📊 Graphiques Radar", "📈 Coordonnées Parallèles", "🌊 Graphiques Stream"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Graphique Radar des Variables Climatiques")
                if 'year' in df.columns:
                    years = sorted(df['year'].unique())
                    if len(years) > 0:
                        selected_year = st.slider(
                            "Sélectionner l'année:",
                            min_value=int(min(years)),
                            max_value=int(max(years)),
                            value=int(max(years))
                        )
                        
                        radar_fig = create_radar_chart(df, selected_year)
                        st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("La colonne 'year' n'est pas disponible dans les données.")
            
            with col2:
                st.markdown("#### 📋 Légende Radar")
                st.markdown("""
                **Axe radial** : Valeurs normalisées (0-1)
                
                **Variables :**
                - **Temp Moy** : Température moyenne
                - **Temp Max** : Température maximale
                - **Temp Min** : Température minimale
                - **Précip** : Précipitations (échelle réduite)
                - **Humidité** : Humidité relative
                - **Vent** : Vitesse du vent
                
                **Interprétation :**
                - Plus l'aire est grande, plus les valeurs sont élevées
                - Comparaison avec la moyenne historique (gris)
                """)
                
                # Comparaison entre années
                st.markdown("#### Comparer deux années")
                if 'year' in df.columns:
                    available_years = sorted(df['year'].unique())
                    if len(available_years) >= 2:
                        # Calculer les indices pour les deux dernières années
                        year1_idx = max(0, len(available_years) - 2)
                        year2_idx = max(0, len(available_years) - 1)
                        
                        year1 = st.selectbox("Année 1", available_years, index=year1_idx)
                        year2 = st.selectbox("Année 2", available_years, index=year2_idx)
                        
                        if year1 != year2:
                            # Créer un radar comparatif
                            fig_compare = go.Figure()
                            
                            for year, color in zip([year1, year2], ['blue', 'red']):
                                year_data = df[df['year'] == year]
                                if len(year_data) > 0:
                                    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
                                    # Vérifier les colonnes manquantes
                                    for col in required_cols:
                                        if col not in year_data.columns:
                                            if col == 'prcp':
                                                year_data[col] = 0
                                            elif col in ['tavg', 'tmax', 'tmin']:
                                                year_data[col] = 20
                                            elif col == 'humidity':
                                                year_data[col] = 50
                                            elif col == 'wind_speed':
                                                year_data[col] = 5
                                    
                                    avg_data = year_data[required_cols].mean()
                                    max_vals = df[required_cols].max()
                                    min_vals = df[required_cols].min()
                                    normalized_data = (avg_data - min_vals) / (max_vals - min_vals)
                                    
                                    fig_compare.add_trace(go.Scatterpolar(
                                        r=[
                                            normalized_data['tavg'],
                                            normalized_data['tmax'],
                                            normalized_data['tmin'],
                                            normalized_data['prcp'] / 100,
                                            normalized_data['humidity'] / 100,
                                            normalized_data['wind_speed'] / 20
                                        ],
                                        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
                                        fill='toself',
                                        name=f'Année {year}',
                                        line_color=color,
                                        opacity=0.5
                                    ))
                            
                            fig_compare.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                title=f'📊 Comparaison {year1} vs {year2}',
                                height=400
                            )
                            
                            st.plotly_chart(fig_compare, use_container_width=True)
        
        with tab2:
            st.markdown("#### Diagramme de Coordonnées Parallèles")
            
            if 'year' in df.columns:
                available_years = sorted(df['year'].unique())
                if available_years:
                    selected_years = st.multiselect(
                        "Sélectionner les années à comparer:",
                        available_years,
                        default=available_years[-min(3, len(available_years)):]  # Maximum 3 dernières années
                    )
                    
                    if selected_years:
                        parallel_fig = create_parallel_coordinates(df, selected_years)
                        st.plotly_chart(parallel_fig, use_container_width=True)
                        
                        st.markdown("**Comment interpréter :**")
                        st.markdown("""
                        - Chaque ligne représente une observation
                        - Les axes verticaux représentent les différentes variables
                        - La couleur montre la valeur de l'année
                        - Les lignes parallèles indiquent des corrélations positives
                        - Les lignes qui se croisent indiquent des corrélations négatives
                        """)
                    else:
                        st.warning("Veuillez sélectionner au moins une année.")
                else:
                    st.warning("Aucune année disponible dans les données.")
            else:
                st.warning("La colonne 'year' n'est pas disponible dans les données.")
        
        with tab3:
            st.markdown("#### Graphique Stream (Courbes Empilées)")
            
            stream_fig = create_stream_graph(df)
            st.plotly_chart(stream_fig, use_container_width=True)
            
            st.markdown("**Explication :**")
            st.markdown("""
            Le graphique stream montre l'évolution des températures moyennes par mois,
            empilées par année. Cela permet de voir :
            
            1. **Tendances saisonnières** : Pattern répétitif chaque année
            2. **Évolution temporelle** : Comment chaque année se compare
            3. **Variabilité** : Largeur de la bande à chaque point
            
            **Utilisations :**
            - Identifier des années exceptionnelles
            - Voir les changements saisonniers
            - Comparer visuellement plusieurs années
            """)
    
    elif page == "🔬 Avancé":
        st.title("🔬 Analyses Avancées et Export")
        
        tab1, tab2, tab3 = st.tabs(["📊 Créateur de Visualisations", "📈 Analyses Temporelles", "💾 Export des Données"])
        
        with tab1:
            st.markdown("#### 🎨 Créateur de Visualisations Personnalisées")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Type de graphique:",
                    ["Ligne Interactive", "Barre Empilée", "Scatter Animé", "Box Plot", "Violon", "Densité"]
                )
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    x_var = st.selectbox("Variable X:", numeric_cols)
                else:
                    x_var = None
            
            with col3:
                if numeric_cols and len(numeric_cols) > 1:
                    y_var = st.selectbox("Variable Y:", numeric_cols, 
                                       index=1 if len(numeric_cols) > 1 else 0)
                else:
                    y_var = None
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                color_options = [None] + categorical_cols
                color_var = st.selectbox(
                    "Couleur par:",
                    color_options
                )
            else:
                color_var = None
            
            if 'year' in df.columns:
                animation_options = [None, 'year', 'month']
                if 'continent' in df.columns:
                    animation_options.append('continent')
                animation_var = st.selectbox(
                    "Animation par:",
                    animation_options
                )
            else:
                animation_var = None
            
            # Options avancées
            with st.expander("⚙️ Options avancées"):
                trendline = st.checkbox("Ajouter une ligne de tendance")
                smoothing = st.checkbox("Lissage des courbes")
                log_scale = st.selectbox("Échelle logarithmique:", [None, "X", "Y", "Les deux"])
            
            if st.button("🔄 Générer la visualisation") and x_var and y_var:
                # Créer le graphique personnalisé
                if chart_type == "Ligne Interactive":
                    fig = px.line(df, x=x_var, y=y_var, color=color_var, 
                                 animation_frame=animation_var,
                                 title=f"{y_var} vs {x_var}")
                    if smoothing:
                        fig.update_traces(line_shape="spline")
                
                elif chart_type == "Barre Empilée":
                    fig = px.bar(df, x=x_var, y=y_var, color=color_var,
                                title=f"{y_var} par {x_var}")
                
                elif chart_type == "Scatter Animé":
                    fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                                    animation_frame=animation_var,
                                    size='prcp' if 'prcp' in df.columns else None,
                                    title=f"Scatter Plot Animé")
                
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_var, y=y_var, color=color_var,
                                title=f"Distribution de {y_var}")
                
                elif chart_type == "Violon":
                    fig = px.violin(df, x=x_var, y=y_var, color=color_var,
                                   title=f"Distribution Densité de {y_var}")
                
                else:  # Densité
                    fig = px.density_heatmap(df, x=x_var, y=y_var,
                                            title=f"Densité {x_var} vs {y_var}")
                
                # Appliquer les options avancées
                if trendline and chart_type in ["Ligne Interactive", "Scatter Animé"]:
                    fig.update_traces(mode='lines+markers')
                
                if log_scale == "X" or log_scale == "Les deux":
                    fig.update_xaxes(type="log")
                if log_scale == "Y" or log_scale == "Les deux":
                    fig.update_yaxes(type="log")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 📈 Analyses Temporelles Avancées")
            
            # Analyse de tendance
            st.markdown("##### Analyse de Tendance")
            
            if 'tavg' in df.columns and 'year' in df.columns:
                # Regression linéaire
                yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
                if len(yearly_avg) > 1:
                    coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                    trend_line = np.poly1d(coeffs)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Pente de tendance", f"{coeffs[0]*10:.3f}°C/décennie")
                        st.metric("Intercept", f"{coeffs[1]:.2f}°C")
                    
                    with col2:
                        correlation = yearly_avg['year'].corr(yearly_avg['tavg'])
                        st.metric("Corrélation", f"{correlation:.3f}")
                        st.metric("R²", f"{correlation**2:.3f}")
                    
                    # Graphique de tendance
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=yearly_avg['year'],
                        y=yearly_avg['tavg'],
                        mode='markers',
                        name='Données',
                        marker=dict(size=10)
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=yearly_avg['year'],
                        y=trend_line(yearly_avg['year']),
                        mode='lines',
                        name=f'Tendance ({coeffs[0]*10:.2f}°C/décennie)',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_trend.update_layout(
                        title='📈 Analyse de Tendance Linéaire',
                        xaxis_title='Année',
                        yaxis_title='Température Moyenne (°C)',
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            # Analyse saisonnière
            st.markdown("##### Analyse Saisonnière")
            
            if 'month' in df.columns and 'tavg' in df.columns:
                seasonal_avg = df.groupby('month')['tavg'].mean().reset_index()
                
                fig_seasonal = px.line_polar(seasonal_avg, r='tavg', theta='month',
                                            line_close=True,
                                            title='🔄 Variation Saisonnière des Températures')
                fig_seasonal.update_traces(fill='toself')
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with tab3:
            st.markdown("#### 💾 Export des Données et Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Export des Données")
                
                # Prévisualisation des données
                st.markdown("**Aperçu des données :**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Options d'export
                export_format = st.selectbox(
                    "Format d'export:",
                    ["CSV", "JSON", "Excel", "Parquet", "PDF Rapport"]
                )
                
                if export_format == "CSV":
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Télécharger CSV",
                        csv,
                        "climate_data_advanced.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Télécharger JSON",
                        json_str,
                        "climate_data_advanced.json",
                        "application/json",
                        key='download-json'
                    )
                
                elif export_format == "Excel":
                    # Pour Excel, on utilise un buffer
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='ClimateData')
                        # Ajouter un onglet avec les statistiques
                        df.describe().to_excel(writer, sheet_name='Statistics')
                    
                    st.download_button(
                        "📥 Télécharger Excel",
                        output.getvalue(),
                        "climate_data_advanced.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download-excel'
                    )
                
                elif export_format == "Parquet":
                    # Pour Parquet, on utilise un buffer temporaire
                    import tempfile
                    try:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                            table = pa.Table.from_pandas(df)
                            pq.write_table(table, tmp.name)
                            
                            with open(tmp.name, 'rb') as f:
                                parquet_data = f.read()
                            
                            st.download_button(
                                "📥 Télécharger Parquet",
                                parquet_data,
                                "climate_data_advanced.parquet",
                                "application/octet-stream",
                                key='download-parquet'
                            )
                    except ImportError:
                        st.error("La bibliothèque pyarrow est requise pour l'export Parquet. Installez-la avec `pip install pyarrow`")
                
                elif export_format == "PDF Rapport":
                    # Section pour générer le PDF
                    st.markdown("##### Génération de Rapport PDF")
                    
                    # Options du rapport
                    with st.expander("⚙️ Options du rapport PDF"):
                        include_temp_chart = st.checkbox("Inclure le graphique d'évolution des températures", value=True)
                        include_heatmap = st.checkbox("Inclure la heatmap", value=True)
                        include_map = st.checkbox("Inclure la carte mondiale", value=True)
                        include_radar = st.checkbox("Inclure le graphique radar", value=True)
                        report_title = st.text_input("Titre du rapport", value="Rapport AgriClima360")
                    
                    if st.button("📄 Générer le Rapport PDF", type="primary"):
                        with st.spinner("⏳ Génération du rapport PDF en cours..."):
                            try:
                                # Préparer les figures pour le PDF
                                temp_fig_pdf = create_temperature_evolution(df) if include_temp_chart else None
                                heatmap_fig_pdf = create_interactive_heatmap(df) if include_heatmap else None
                                map_fig_pdf = create_animated_temperature_map(df) if include_map else None
                                
                                # Pour le radar, on prend l'année la plus récente
                                if include_radar and 'year' in df.columns:
                                    radar_year = df['year'].max()
                                    radar_fig_pdf = create_radar_chart(df, radar_year)
                                else:
                                    radar_fig_pdf = None
                                
                                # Générer le PDF
                                pdf_bytes = generate_pdf_report(
                                    df=df,
                                    kpis=kpis,
                                    temp_fig=temp_fig_pdf,
                                    heatmap_fig=heatmap_fig_pdf,
                                    map_fig=map_fig_pdf,
                                    radar_fig=radar_fig_pdf
                                )
                                
                                # Nom du fichier avec timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"{report_title.replace(' ', '_')}_{timestamp}.pdf"
                                
                                # Afficher le bouton de téléchargement
                                st.success("✅ Rapport PDF généré avec succès !")
                                
                                # Bouton de téléchargement
                                st.download_button(
                                    label="📥 Télécharger le Rapport PDF",
                                    data=pdf_bytes,
                                    file_name=filename,
                                    mime="application/pdf",
                                    key='download-pdf-report'
                                )
                                
                                # Aperçu du contenu
                                with st.expander("📋 Aperçu du contenu du rapport"):
                                    st.markdown("""
                                    **Sections incluses dans le rapport:**
                                    
                                    1. **Page de titre** avec métadonnées
                                    2. **Résumé exécutif** avec contexte d'analyse
                                    3. **Indicateurs Clés (KPIs)** sous forme de tableau
                                    4. **Graphiques climatiques** (selon vos sélections)
                                    5. **Statistiques détaillées** par variable
                                    6. **Recommandations** pour l'agriculture
                                    
                                    **Caractéristiques:**
                                    • Format: PDF A4, 3-4 pages
                                    • Style professionnel avec mise en forme
                                    • Données actualisées selon vos filtres
                                    • Génération automatique avec timestamp
                                    """)
                                
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la génération du PDF: {str(e)}")
                                st.info("💡 Assurez-vous d'avoir installé les bibliothèques requises: `pip install fpdf kaleido`")
            
            with col2:
                st.markdown("##### Export des Visualisations")
                
                # Options pour exporter les graphiques
                chart_to_export = st.selectbox(
                    "Graphique à exporter:",
                    ["Évolution des Températures", "Carte Animée", "Graphique 3D", 
                     "Radar Chart", "Matrice de Corrélation"]
                )
                
                format_img = st.selectbox(
                    "Format d'image:",
                    ["PNG", "JPEG", "SVG", "PDF"]
                )
                
                if st.button("🖼️ Générer l'image"):
                    # Créer le graphique sélectionné
                    if chart_to_export == "Évolution des Températures":
                        fig = create_temperature_evolution(df)
                    elif chart_to_export == "Carte Animée":
                        fig = create_animated_temperature_map(df)
                    elif chart_to_export == "Graphique 3D":
                        fig = create_3d_scatter_plot(df)
                    elif chart_to_export == "Radar Chart":
                        fig = create_radar_chart(df, df['year'].max() if 'year' in df.columns else None)
                    else:
                        fig = create_correlation_matrix_interactive(df)
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bouton de téléchargement (note: Plotly ne permet pas le téléchargement direct en SVG/PDF)
                    st.info(f"Pour sauvegarder en {format_img}, utilisez l'icône de capture dans la barre d'outils du graphique.")
                
                st.markdown("---")
                st.markdown("##### Rapport Automatique")
                
                # Note: Cette section est maintenant déplacée dans l'option "PDF Rapport"
                st.info("La génération de rapport PDF est maintenant disponible dans l'option 'PDF Rapport' ci-dessus.")
    
    # Footer avec informations
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌍 AgriClima360 - Dashboard Climatique Avancé avec Animations Interactives</p>
            <p style='font-size: 0.8em; color: gray;'>
                Données fournies par NOAA National Centers for Environmental Information | 
                <strong>Fonctionnalités avancées</strong> : Animations, 3D, Carte interactive, Graphiques radar, Export PDF
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
