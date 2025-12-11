# =============================================================
# AGRICLIMA360 - Application Streamlit avec données NOAA API
# Visualisations climatiques interactives AVEC ANIMATIONS
# et visualisation de données massives
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
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# IMPORT DES LIBRAIRIES DE VISUALISATION MASSIVE
# =============================================================
try:
    import dask.dataframe as dd
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader import reductions
    from datashader.colors import inferno, viridis
    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread
    import hvplot.pandas
    import hvplot.dask
    import panel as pn
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256, Inferno256
    from bokeh.embed import components
    from bokeh.resources import CDN
    
    hv.extension('bokeh')
    pn.extension()
    DATA_VIZ_ENABLED = True
    st.success("✅ Visualisation de données massives activée (Dask + Datashader)")
except ImportError as e:
    DATA_VIZ_ENABLED = False
    st.warning(f"⚠️ Visualisation de données massives désactivée: {e}")

# =============================================================
# CONFIGURATION
# =============================================================

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
NOAA_TOKEN = st.secrets.get("NOAA_TOKEN", "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM")

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
    df['week'] = df['date'].dt.isocalendar().week
    
    # Conversion des températures (de dixièmes de degrés Celsius)
    if 'value' in df.columns:
        # Les températures NOAA sont en dixièmes de degrés
        temp_types = ['TMAX', 'TMIN', 'TAVG']
        df.loc[df['datatype'].isin(temp_types), 'value'] = df.loc[df['datatype'].isin(temp_types), 'value'] / 10
        
        # Les précipitations sont en dixièmes de mm
        df.loc[df['datatype'] == 'PRCP', 'value'] = df.loc[df['datatype'] == 'PRCP', 'value'] / 10
    
    # Pivoter pour avoir les différents types de données en colonnes
    df_pivot = df.pivot_table(
        index=['date', 'year', 'month', 'day', 'day_of_year', 'week', 'station'],
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
        'WSF2': 'wind_fastest',
        'WDF2': 'wind_direction',
        'WSF5': 'wind_gust'
    }
    # Renommer uniquement les colonnes existantes
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=existing_columns)
    
    # Calculer tavg si manquant
    if 'tavg' not in df_pivot.columns and 'tmax' in df_pivot.columns and 'tmin' in df_pivot.columns:
        df_pivot['tavg'] = (df_pivot['tmax'] + df_pivot['tmin']) / 2
    
    # Ajouter des données simulées pour les visualisations avancées
    n_rows = len(df_pivot)
    df_pivot['humidity'] = np.random.uniform(30, 90, n_rows)
    df_pivot['wind_speed'] = np.random.uniform(0, 20, n_rows)
    df_pivot['solar_radiation'] = np.random.uniform(100, 800, n_rows)
    df_pivot['pressure'] = np.random.uniform(980, 1030, n_rows)
    df_pivot['evapotranspiration'] = np.random.uniform(0, 10, n_rows)
    df_pivot['soil_moisture'] = np.random.uniform(10, 80, n_rows)
    df_pivot['continent'] = np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania'], n_rows)
    df_pivot['country'] = np.random.choice(['USA', 'Canada', 'France', 'Germany', 'China', 'India', 'Brazil', 'Australia'], n_rows)
    df_pivot['lat'] = 40.0 + np.random.uniform(-30, 30, n_rows)
    df_pivot['lon'] = -100.0 + np.random.uniform(-60, 60, n_rows)
    df_pivot['elevation'] = np.random.uniform(0, 3000, n_rows)
    
    # Ajouter des tendances temporelles
    df_pivot['warming_trend'] = 0.01 * (df_pivot['year'] - 2000)
    df_pivot['tavg_trend'] = df_pivot['tavg'] + df_pivot['warming_trend']
    
    return df_pivot

def generate_massive_sample_data(n_points=1000000):
    """Génère des données de démonstration massives pour tester Dask/Datashader."""
    st.info(f"🧪 Génération de {n_points:,} points de données pour les tests...")
    
    # Créer un DataFrame Dask
    n_partitions = 10
    chunk_size = n_points // n_partitions
    
    def generate_chunk(i):
        """Génère un chunk de données."""
        n = min(chunk_size, n_points - i * chunk_size)
        
        dates = pd.date_range('2000-01-01', '2023-12-31', periods=n)
        
        data = {
            'date': np.random.choice(dates, n),
            'year': np.random.randint(2000, 2024, n),
            'month': np.random.randint(1, 13, n),
            'day': np.random.randint(1, 29, n),
            'station': [f'ST{i:04d}' for i in np.random.randint(1, 1000, n)],
            'tavg': 15 + 10 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'tmax': 20 + 12 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'tmin': 10 + 8 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'prcp': np.random.exponential(5, n),
            'humidity': np.random.uniform(30, 90, n),
            'wind_speed': np.random.exponential(5, n),
            'solar_radiation': np.random.uniform(100, 800, n),
            'pressure': np.random.normal(1013, 10, n),
            'lat': np.random.uniform(-90, 90, n),
            'lon': np.random.uniform(-180, 180, n),
            'elevation': np.random.exponential(500, n),
            'continent': np.random.choice(['NA', 'EU', 'AS', 'AF', 'SA', 'OC'], n),
            'biome': np.random.choice(['Forest', 'Grassland', 'Desert', 'Tundra', 'Aquatic'], n)
        }
        
        return pd.DataFrame(data)
    
    # Créer un DataFrame Dask
    if DATA_VIZ_ENABLED:
        # Créer une liste de DataFrames pandas
        dfs = [generate_chunk(i) for i in range(n_partitions)]
        
        # Convertir en DataFrame Dask
        ddf = dd.from_pandas(pd.concat(dfs, ignore_index=True), npartitions=n_partitions)
        
        # Optimiser les types de données
        ddf['date'] = dd.to_datetime(ddf['date'])
        ddf['station'] = ddf['station'].astype('category')
        ddf['continent'] = ddf['continent'].astype('category')
        ddf['biome'] = ddf['biome'].astype('category')
        
        st.success(f"✅ {n_points:,} points générés avec Dask ({n_partitions} partitions)")
        return ddf
    else:
        # Version pandas (plus lente)
        df = generate_chunk(0)
        for i in range(1, n_partitions):
            df = pd.concat([df, generate_chunk(i)], ignore_index=True)
        
        st.success(f"✅ {len(df):,} points générés avec Pandas")
        return df

def generate_enhanced_sample_data(n_points=100000):
    """Génère des données de démonstration enrichies."""
    st.warning("🔧 Données de démonstration - Configurez votre token NOAA pour des données réelles.")
    
    if DATA_VIZ_ENABLED and n_points > 100000:
        return generate_massive_sample_data(min(n_points, 500000))
    
    dates = pd.date_range('2000-01-01', '2023-12-31', periods=n_points)
    
    data = {
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'day_of_year': dates.dayofyear,
        'week': dates.isocalendar().week,
        'station': [f'ST{i:04d}' for i in np.random.randint(1, 100, n_points)],
        'tavg': 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'tmax': 20 + 12 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'tmin': 10 + 8 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'prcp': np.random.exponential(5, n_points),
        'humidity': np.random.uniform(30, 90, n_points),
        'wind_speed': np.random.exponential(5, n_points),
        'solar_radiation': np.random.uniform(100, 800, n_points),
        'pressure': np.random.normal(1013, 10, n_points),
        'evapotranspiration': np.random.uniform(0, 10, n_points),
        'soil_moisture': np.random.uniform(10, 80, n_points),
        'continent': np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania'], n_points),
        'country': np.random.choice(['USA', 'Canada', 'France', 'Germany', 'China', 'India', 'Brazil', 'Australia'], n_points),
        'lat': np.random.uniform(-90, 90, n_points),
        'lon': np.random.uniform(-180, 180, n_points),
        'elevation': np.random.exponential(500, n_points)
    }
    
    return pd.DataFrame(data)

def compute_kpis(df):
    """Calcule les indicateurs clés avancés."""
    kpis = {}
    
    if not df.empty:
        # Utiliser Dask pour les calculs si disponible
        if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
            with ProgressBar():
                kpis["temp_moy"] = df["tavg"].mean().compute() if "tavg" in df.columns else 0
                kpis["pluie_totale"] = df["prcp"].sum().compute() if "prcp" in df.columns else 0
                kpis["nb_annees"] = df["year"].nunique().compute()
                kpis["nb_stations"] = df["station"].nunique().compute() if "station" in df.columns else 0
                kpis["temp_max"] = df["tmax"].max().compute() if "tmax" in df.columns else 0
                kpis["temp_min"] = df["tmin"].min().compute() if "tmin" in df.columns else 0
                kpis["humidite_moy"] = df["humidity"].mean().compute() if "humidity" in df.columns else 65
                kpis["solar_avg"] = df["solar_radiation"].mean().compute() if "solar_radiation" in df.columns else 0
                kpis["wind_avg"] = df["wind_speed"].mean().compute() if "wind_speed" in df.columns else 0
                kpis["nb_points"] = len(df)
        else:
            # Version pandas
            kpis["temp_moy"] = df["tavg"].mean() if "tavg" in df.columns else 0
            kpis["pluie_totale"] = df["prcp"].sum() if "prcp" in df.columns else 0
            kpis["nb_annees"] = df["year"].nunique()
            kpis["nb_stations"] = df["station"].nunique() if "station" in df.columns else 0
            kpis["temp_max"] = df["tmax"].max() if "tmax" in df.columns else 0
            kpis["temp_min"] = df["tmin"].min() if "tmin" in df.columns else 0
            kpis["humidite_moy"] = df["humidity"].mean() if "humidity" in df.columns else 65
            kpis["solar_avg"] = df["solar_radiation"].mean() if "solar_radiation" in df.columns else 0
            kpis["wind_avg"] = df["wind_speed"].mean() if "wind_speed" in df.columns else 0
            kpis["nb_points"] = len(df)
        
        # Calcul de la tendance de température
        if "tavg" in df.columns and kpis["nb_annees"] > 1:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                yearly_avg = df.groupby('year')['tavg'].mean().compute().reset_index()
            else:
                yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
            
            if len(yearly_avg) > 1:
                coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                kpis["temp_trend"] = coeffs[0] * 100  # °C par siècle
                kpis["temp_trend_decade"] = coeffs[0] * 10  # °C par décennie
            else:
                kpis["temp_trend"] = 0
                kpis["temp_trend_decade"] = 0
        else:
            kpis["temp_trend"] = 0
            kpis["temp_trend_decade"] = 0
            
        # Calcul de la variabilité
        if "tavg" in df.columns and kpis["nb_annees"] > 1:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                kpis["variability"] = df.groupby('year')['tavg'].std().mean().compute()
            else:
                kpis["variability"] = df.groupby('year')['tavg'].std().mean()
        else:
            kpis["variability"] = 0
            
        # Calcul des canicules
        if "tmax" in df.columns and kpis["nb_points"] > 0:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                heatwave_count = (df['tmax'] > 30).sum().compute()
                kpis["heatwaves"] = (heatwave_count / kpis["nb_points"]) * 100
            else:
                kpis["heatwaves"] = (df['tmax'] > 30).sum() / kpis["nb_points"] * 100
        else:
            kpis["heatwaves"] = 0
            
        # Calcul du risque de sécheresse
        if "prcp" in df.columns and kpis["nb_points"] > 0:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                drought_count = (df['prcp'] < 5).sum().compute()
                kpis["drought_risk"] = (drought_count / kpis["nb_points"]) * 100
            else:
                kpis["drought_risk"] = (df['prcp'] < 5).sum() / kpis["nb_points"] * 100
        else:
            kpis["drought_risk"] = 0
            
        # Nombre de continents
        if "continent" in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                kpis["continents"] = df["continent"].nunique().compute()
            else:
                kpis["continents"] = df["continent"].nunique()
        else:
            kpis["continents"] = 1
    
    return kpis

# =============================================================
# 3. FONCTIONS DE VISUALISATION MASSIVES (DASK + DATASHADER)
# =============================================================

def create_datashader_plot(df, x_col='lon', y_col='lat', color_col='tavg', 
                          title='Carte Thermique avec Datashader', width=800, height=600):
    """Crée une visualisation Datashader pour des millions de points."""
    if not DATA_VIZ_ENABLED:
        st.warning("Datashader non disponible")
        return None
    
    try:
        # Convertir en DataFrame pandas si c'est un Dask DataFrame
        if isinstance(df, dd.DataFrame):
            # Échantillonner pour Datashader
            sample_size = min(1000000, len(df))
            df_sample = df.sample(frac=sample_size/len(df)).compute() if len(df) > sample_size else df.compute()
        else:
            df_sample = df.sample(min(1000000, len(df))) if len(df) > 1000000 else df
        
        # Créer le canvas Datashader
        canvas = ds.Canvas(plot_width=width, plot_height=height)
        
        # Agréger les points
        agg = canvas.points(df_sample, x_col, y_col, ds.mean(color_col))
        
        # Appliquer la colormap
        img = tf.shade(agg, cmap=viridis, how='log')
        img = tf.set_background(img, 'black')
        
        # Convertir en image
        img_pil = img.to_pil()
        
        return img_pil
        
    except Exception as e:
        st.error(f"Erreur Datashader: {e}")
        return None

def create_holoviews_datashader(df, x_col='date', y_col='tavg', color_col='prcp',
                               title='Time Series avec Datashader'):
    """Crée une visualisation HoloViews avec Datashader."""
    if not DATA_VIZ_ENABLED:
        return None
    
    try:
        # Échantillonner si nécessaire
        if isinstance(df, dd.DataFrame):
            df_plot = df.sample(frac=0.1).compute() if len(df) > 100000 else df.compute()
        else:
            df_plot = df.sample(min(100000, len(df)))
        
        # Créer le scatter plot
        scatter = hv.Scatter(df_plot, x_col, y_col).opts(
            width=800,
            height=400,
            title=title,
            color=color_col,
            cmap='viridis',
            colorbar=True,
            tools=['hover']
        )
        
        # Appliquer Datashader
        shaded = dynspread(datashade(scatter, cmap=viridis))
        
        return shaded
        
    except Exception as e:
        st.error(f"Erreur HoloViews: {e}")
        return None

def create_dask_histogram(df, column='tavg', bins=100, title='Distribution avec Dask'):
    """Crée un histogramme avec Dask pour de grandes données."""
    if not DATA_VIZ_ENABLED or not isinstance(df, dd.DataFrame):
        # Version pandas
        fig = px.histogram(df, x=column, nbins=bins, title=title)
        return fig
    
    try:
        with ProgressBar():
            # Calculer l'histogramme avec Dask
            hist, edges = da.histogram(df[column].to_dask_array(), bins=bins, range=[df[column].min().compute(), df[column].max().compute()])
            hist_values = hist.compute()
            edges_values = edges.compute()
        
        # Créer le graphique
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=edges_values[:-1],
            y=hist_values,
            width=np.diff(edges_values),
            marker_color='royalblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title='Fréquence',
            bargap=0.05,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur Dask histogram: {e}")
        return None

def create_massive_scatter(df, x_col='tavg', y_col='prcp', color_col='year',
                          title='Scatter Plot Massif', point_size=2):
    """Crée un scatter plot pour des millions de points."""
    if len(df) > 100000 and DATA_VIZ_ENABLED:
        # Utiliser Datashader pour les grandes données
        if isinstance(df, dd.DataFrame):
            df_sample = df.sample(frac=0.1).compute() if len(df) > 1000000 else df.compute()
        else:
            df_sample = df.sample(min(100000, len(df)))
        
        # Créer avec Datashader
        canvas = ds.Canvas(plot_width=800, plot_height=500)
        agg = canvas.points(df_sample, x_col, y_col, ds.mean(color_col) if color_col else ds.count())
        img = tf.shade(agg, cmap=viridis)
        
        # Convertir en figure Plotly
        img_array = np.array(img.to_pil())
        fig = px.imshow(img_array, title=f"{title} (Datashader - {len(df):,} points)")
        return fig
    else:
        # Version Plotly normale
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        
        fig = px.scatter(df_sample, x=x_col, y=y_col, color=color_col,
                        title=f"{title} ({len(df_sample):,} points échantillonnés)",
                        opacity=0.6,
                        hover_data=['date', 'station'] if 'date' in df.columns and 'station' in df.columns else None)
        fig.update_traces(marker=dict(size=point_size))
        return fig

def create_spatial_heatmap(df, title='Carte de Chaleur Spatiale'):
    """Crée une carte de chaleur spatiale avec Datashader."""
    if not DATA_VIZ_ENABLED or 'lat' not in df.columns or 'lon' not in df.columns:
        return None
    
    try:
        # Préparer les données
        if isinstance(df, dd.DataFrame):
            df_spatial = df[['lat', 'lon', 'tavg']].dropna().compute()
        else:
            df_spatial = df[['lat', 'lon', 'tavg']].dropna()
        
        # Créer le canvas
        canvas = ds.Canvas(plot_width=800, plot_height=400)
        
        # Agréger
        agg = canvas.points(df_spatial, 'lon', 'lat', ds.mean('tavg'))
        
        # Créer l'image
        img = tf.shade(agg, cmap=inferno, how='log')
        img = tf.set_background(img, 'white')
        
        return img.to_pil()
        
    except Exception as e:
        st.error(f"Erreur carte de chaleur: {e}")
        return None

def create_time_series_aggregation(df, time_col='date', value_col='tavg', 
                                  freq='M', title='Série Temporelle Agrégée'):
    """Crée une série temporelle agrégée avec Dask."""
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        try:
            with ProgressBar():
                # Agrégation temporelle avec Dask
                df['date'] = dd.to_datetime(df[time_col])
                df_resampled = df.set_index('date').resample(freq).mean()[value_col].compute()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled.values,
                mode='lines',
                name=value_col,
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"{title} (Dask - {len(df):,} points)",
                xaxis_title='Date',
                yaxis_title=value_col,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erreur Dask resample: {e}")
    
    # Version pandas
    if 'date' in df.columns:
        if isinstance(df, dd.DataFrame):
            df_pd = df.compute()
        else:
            df_pd = df.copy()
        
        df_pd['date'] = pd.to_datetime(df_pd[time_col])
        df_pd.set_index('date', inplace=True)
        df_resampled = df_pd.resample(freq).mean()[value_col]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled.values,
            mode='lines',
            name=value_col,
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f"{title} ({len(df):,} points)",
            xaxis_title='Date',
            yaxis_title=value_col,
            height=400
        )
        
        return fig
    
    return None

# =============================================================
# 4. FONCTIONS DE VISUALISATION STANDARD
# =============================================================

def create_temperature_evolution(df):
    """Crée le graphique d'évolution des températures avec animation."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    # Utiliser Dask pour les calculs si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            yearly_data = df.groupby('year').agg({
                'tavg': 'mean',
                'tmax': 'max',
                'tmin': 'min'
            }).compute().reset_index()
    else:
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
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            monthly_prcp = df.groupby(['year', 'month'])['prcp'].sum().compute().reset_index()
    else:
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
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            yearly_avg = df.groupby(['year', 'continent']).agg({
                'tavg': 'mean',
                'tmax': 'max',
                'tmin': 'min',
                'prcp': 'sum',
                'lat': 'mean',
                'lon': 'mean'
            }).compute().reset_index()
    else:
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
    
    sample_size = min(5000, len(df))
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        df_sample = df.sample(frac=sample_size/len(df)).compute() if len(df) > sample_size else df.compute()
    else:
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
    
    fig = px.scatter_3d(df_sample,
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
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            pivot_data = df.pivot_table(index='month', columns='year', values='tavg', aggfunc='mean').compute()
    else:
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

# =============================================================
# 5. INTERFACE STREAMLIT AVANCÉE
# =============================================================

def main():
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("🌾 AgriClima360")
        st.markdown("### Dashboard Climatique Avancé")
        st.markdown("*Visualisation de données massives*" if DATA_VIZ_ENABLED else "*Mode standard*")
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Sélection de la source de données
        data_source = st.radio(
            "Source de données:",
            ["API NOAA (Réelles)", "Démonstration", "Données Massives (Test)"]
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
        pages = [
            "🏠 Vue d'ensemble", 
            "📈 Analyses Animées", 
            "🌐 Visualisations 3D", 
            "🗺️ Carte Animée",
            "🚀 Données Massives",
            "🔬 Avancé", 
            "🎯 Radar & Parallèles"
        ]
        
        page = st.radio("Sections:", pages)
        
        st.markdown("---")
        
        # Filtres
        st.header("🎛️ Filtres")
        
        # Ces filtres seront appliqués après chargement
        year_filter = st.empty()
        continent_filter = st.empty()
        data_size_filter = st.empty()
        
        st.markdown("---")
        
        # Contrôles d'animation
        st.header("🎬 Contrôles d'Animation")
        animation_speed = st.slider("Vitesse d'animation:", 100, 2000, 500, 100)
        auto_play = st.checkbox("Lecture automatique", value=True)
        
        # Options de visualisation massive
        if DATA_VIZ_ENABLED:
            st.header("🚀 Options Massives")
            enable_dask = st.checkbox("Utiliser Dask", value=True)
            sample_size = st.selectbox(
                "Taille de l'échantillon:",
                ["1K", "10K", "100K", "1M", "10M", "Complet"],
                index=2
            )
        
        # Boutons d'export
        st.header("💾 Export")
        export_format = st.selectbox("Format d'export:", ["CSV", "Parquet", "JSON", "Excel"])
    
    # Chargement des données
    with st.spinner("⏳ Chargement des données..."):
        if data_source == "API NOAA (Réelles)":
            if NOAA_TOKEN == "YOUR_TOKEN_HERE" or NOAA_TOKEN == "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM":
                st.error("❌ Token NOAA non configuré. Créez un fichier `.streamlit/secrets.toml` avec:\n```toml\nNOAA_TOKEN = 'votre_token'\n```")
                df = generate_enhanced_sample_data(50000)
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
                
        elif data_source == "Démonstration":
            df = generate_enhanced_sample_data(50000)
            
        else:  # Données Massives
            if DATA_VIZ_ENABLED:
                # Définir la taille des données
                size_map = {"1K": 1000, "10K": 10000, "100K": 100000, "1M": 1000000, "10M": 10000000}
                n_points = size_map.get(sample_size, 100000)
                
                if sample_size == "Complet" and enable_dask:
                    # Générer des données massives avec Dask
                    df = generate_massive_sample_data(1000000)
                else:
                    df = generate_enhanced_sample_data(n_points)
                    
                if enable_dask and not isinstance(df, dd.DataFrame):
                    # Convertir en Dask DataFrame
                    n_partitions = max(1, len(df) // 100000)
                    df = dd.from_pandas(df, npartitions=n_partitions)
                    st.success(f"✅ Converti en Dask DataFrame ({n_partitions} partitions)")
            else:
                df = generate_enhanced_sample_data(100000)
    
    # Vérification des données
    if df.empty:
        st.error("❌ Aucune donnée disponible. Vérifiez vos paramètres.")
        return
    
    # Afficher les informations sur les données
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        st.sidebar.info(f"📊 Données Dask: {len(df):,} points, {df.npartitions} partitions")
    else:
        st.sidebar.info(f"📊 Données: {len(df):,} points")
    
    # Calcul des KPIs
    with st.spinner("📊 Calcul des indicateurs..."):
        kpis = compute_kpis(df)
    
    # Filtres dans la sidebar
    with st.sidebar:
        if 'year' in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                with ProgressBar():
                    years = sorted(df['year'].unique().compute())
            else:
                years = sorted(df['year'].unique())
                
            if len(years) > 0:
                selected_years = year_filter.slider(
                    "Période:",
                    int(min(years)),
                    int(max(years)),
                    (int(min(years)), int(max(years)))
                )
                # Appliquer le filtre
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
                else:
                    df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
        
        if 'continent' in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                with ProgressBar():
                    continents = ['Tous'] + sorted(df['continent'].unique().compute().tolist())
            else:
                continents = ['Tous'] + sorted(df['continent'].unique().tolist())
                
            selected_continent = continent_filter.selectbox(
                "Continent:",
                continents
            )
            if selected_continent != 'Tous':
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df[df['continent'] == selected_continent]
                else:
                    df = df[df['continent'] == selected_continent]
        
        # Filtre de taille de données pour les démos
        if data_source == "Données Massives (Test)":
            viz_size = data_size_filter.slider(
                "Points à visualiser:",
                1000, 1000000, 100000, 1000,
                help="Réduire pour améliorer les performances"
            )
            if len(df) > viz_size:
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df.sample(frac=viz_size/len(df))
                else:
                    df = df.sample(min(viz_size, len(df)))
    
    # Vérifier à nouveau si le dataframe n'est pas vide après filtrage
    if len(df) == 0:
        st.error("❌ Aucune donnée disponible après filtrage. Ajustez vos critères.")
        return
    
    # =============================================================
    # PAGES AVEC VISUALISATIONS
    # =============================================================
    
    if page == "🏠 Vue d'ensemble":
        st.title("🌍 AgriClima360 - Dashboard Climatique Avancé")
        st.markdown("### Visualisations interactives avec animations")
        
        if DATA_VIZ_ENABLED:
            st.info(f"🚀 Mode données massives activé: {len(df):,} points de données")
        
        # KPIs en ligne
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🌡️ Température Moy.",
                f"{kpis.get('temp_moy', 0):.1f}°C",
                f"{kpis.get('temp_trend_decade', 0):+.2f}°C/décennie"
            )
        
        with col2:
            st.metric(
                "💧 Précipitations",
                f"{kpis.get('pluie_totale', 0):,.0f} mm",
                f"{kpis.get('nb_points', 0):,} points"
            )
        
        with col3:
            st.metric(
                "⚠️ Canicules",
                f"{kpis.get('heatwaves', 0):.1f}%",
                f"Stations: {kpis.get('nb_stations', 0)}"
            )
        
        with col4:
            st.metric(
                "🌞 Radiation Solaire",
                f"{kpis.get('solar_avg', 0):.0f} W/m²",
                f"Vent: {kpis.get('wind_avg', 0):.1f} m/s"
            )
        
        with col5:
            if "continents" in kpis:
                st.metric("🌐 Continents", f"{kpis.get('continents', 1)}", f"Années: {kpis.get('nb_annees', 1)}")
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Évolution Temporelle")
            st.plotly_chart(
                create_temperature_evolution(df),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 💧 Précipitations")
            st.plotly_chart(
                create_precipitation_chart(df),
                use_container_width=True
            )
        
        # Visualisation massive si activée
        if DATA_VIZ_ENABLED and len(df) > 100000:
            st.markdown("---")
            st.markdown("#### 🚀 Visualisation Massive")
            
            tab1, tab2 = st.tabs(["Carte Thermique", "Distribution"])
            
            with tab1:
                st.markdown("##### Carte Thermique avec Datashader")
                if 'lat' in df.columns and 'lon' in df.columns:
                    img = create_datashader_plot(df, title=f"Carte Thermique ({len(df):,} points)")
                    if img:
                        st.image(img, caption="Carte thermique des températures moyennes", use_column_width=True)
                else:
                    st.warning("Données spatiales nécessaires pour la carte thermique")
            
            with tab2:
                st.markdown("##### Distribution des Températures")
                fig_hist = create_dask_histogram(df, column='tavg', title=f"Distribution des Températures ({len(df):,} points)")
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    elif page == "🚀 Données Massives":
        st.title("🚀 Visualisation de Données Massives")
        
        if not DATA_VIZ_ENABLED:
            st.error("❌ Les packages de visualisation massive ne sont pas installés.")
            st.info("Installez-les avec: `pip install dask datashader holoviews hvplot panel bokeh`")
            return
        
        st.info(f"📊 Traitement de {len(df):,} points de données avec Dask et Datashader")
        
        # Sélection de visualisation
        viz_type = st.selectbox(
            "Type de visualisation:",
            ["Scatter Plot Massif", "Carte Thermique Spatiale", "Série Temporelle Agrégée", 
             "Distribution Dask", "Holoviews + Datashader", "Comparaison de Performances"]
        )
        
        if viz_type == "Scatter Plot Massif":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_var = st.selectbox("Variable X:", ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed'])
            with col2:
                y_var = st.selectbox("Variable Y:", ['prcp', 'tavg', 'humidity', 'wind_speed', 'solar_radiation'])
            with col3:
                color_var = st.selectbox("Couleur:", ['year', 'month', 'continent', None])
            
            point_size = st.slider("Taille des points:", 1, 10, 2)
            
            fig = create_massive_scatter(df, x_col=x_var, y_col=y_var, color_col=color_var,
                                       title=f"Scatter Plot: {y_var} vs {x_var}", point_size=point_size)
            st.plotly_chart(fig, use_container_width=True)
            
            # Informations sur les performances
            if isinstance(df, dd.DataFrame):
                st.info(f"✅ Graphique généré à partir de {len(df):,} points avec Dask")
        
        elif viz_type == "Carte Thermique Spatiale":
            st.markdown("#### 🌍 Carte de Chaleur Spatiale")
            
            if 'lat' in df.columns and 'lon' in df.columns:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    img = create_spatial_heatmap(df)
                    if img:
                        st.image(img, caption="Carte de chaleur spatiale des températures", use_column_width=True)
                
                with col2:
                    st.markdown("**Paramètres :**")
                    st.metric("Points", f"{len(df):,}")
                    if isinstance(df, dd.DataFrame):
                        st.metric("Partitions", df.npartitions)
                    
                    # Options d'affichage
                    color_map = st.selectbox("Colormap:", ['inferno', 'viridis', 'plasma', 'magma'])
                    point_size = st.slider("Taille:", 1, 20, 5)
                    
                    if st.button("🔄 Regénérer"):
                        st.rerun()
            else:
                st.warning("Les colonnes 'lat' et 'lon' sont nécessaires pour la carte spatiale")
        
        elif viz_type == "Série Temporelle Agrégée":
            st.markdown("#### 📈 Série Temporelle Agrégée")
            
            col1, col2 = st.columns(2)
            
            with col1:
                value_col = st.selectbox("Variable:", ['tavg', 'tmax', 'tmin', 'prcp', 'humidity'])
            with col2:
                freq = st.selectbox("Fréquence:", ['D', 'W', 'M', 'Q', 'Y'])
            
            fig = create_time_series_aggregation(df, value_col=value_col, freq=freq,
                                                title=f"Série Temporelle de {value_col}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques temporelles
            if 'date' in df.columns:
                if isinstance(df, dd.DataFrame):
                    with ProgressBar():
                        date_range = df['date'].min().compute(), df['date'].max().compute()
                        days = (date_range[1] - date_range[0]).days
                else:
                    date_range = df['date'].min(), df['date'].max()
                    days = (date_range[1] - date_range[0]).days
                
                st.metric("Période", f"{date_range[0].date()} à {date_range[1].date()}")
                st.metric("Durée", f"{days} jours")
        
        elif viz_type == "Distribution Dask":
            st.markdown("#### 📊 Distributions avec Dask")
            
            col1, col2 = st.columns(2)
            
            with col1:
                variable = st.selectbox("Variable à analyser:", 
                                       ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed'])
            with col2:
                bins = st.slider("Nombre de bins:", 10, 500, 100)
            
            fig = create_dask_histogram(df, column=variable, bins=bins,
                                       title=f"Distribution de {variable}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            if isinstance(df, dd.DataFrame):
                with ProgressBar():
                    stats = df[variable].describe().compute()
                st.dataframe(stats, use_container_width=True)
        
        elif viz_type == "Holoviews + Datashader":
            st.markdown("#### 🎨 HoloViews avec Datashader")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("Axe X:", ['date', 'tavg', 'prcp', 'humidity'])
            with col2:
                y_var = st.selectbox("Axe Y:", ['tavg', 'prcp', 'humidity', 'wind_speed'])
            
            plot = create_holoviews_datashader(df, x_col=x_var, y_col=y_var,
                                             title=f"{y_var} vs {x_var}")
            if plot:
                # Convertir HoloViews en HTML pour Streamlit
                hv.save(plot, 'temp_plot.html')
                with open('temp_plot.html', 'r') as f:
                    html = f.read()
                
                # Afficher dans Streamlit
                components.html(html, height=500)
            else:
                st.warning("Impossible de créer la visualisation HoloViews")
        
        elif viz_type == "Comparaison de Performances":
            st.markdown("#### ⚡ Comparaison de Performances")
            
            # Test de performance
            st.markdown("**Benchmark des opérations :**")
            
            if st.button("🚀 Lancer le benchmark"):
                with st.spinner("Exécution des tests..."):
                    # Test 1: GroupBy
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result1 = df.groupby('year')['tavg'].mean().compute()
                    else:
                        result1 = df.groupby('year')['tavg'].mean()
                    time1 = time.time() - start
                    
                    # Test 2: Filtre
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result2 = df[df['tavg'] > 20].compute()
                    else:
                        result2 = df[df['tavg'] > 20]
                    time2 = time.time() - start
                    
                    # Test 3: Statistiques
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result3 = df['tavg'].describe().compute()
                    else:
                        result3 = df['tavg'].describe()
                    time3 = time.time() - start
                
                # Afficher les résultats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("GroupBy", f"{time1:.2f}s", f"{len(result1)} groupes")
                with col2:
                    st.metric("Filtre", f"{time2:.2f}s", f"{len(result2):,} points")
                with col3:
                    st.metric("Statistiques", f"{time3:.2f}s", "8 métriques")
                
                # Recommandations
                st.markdown("**Recommandations :**")
                if isinstance(df, dd.DataFrame):
                    st.success("✅ Dask est activé - Optimisé pour les grandes données")
                    st.info(f"Partitions: {df.npartitions}, Points: {len(df):,}")
                else:
                    st.warning("⚠️ Pandas seul - Pensez à activer Dask pour +100K points")
    
    elif page == "📈 Analyses Animées":
        st.title("📊 Analyses avec Animations")
        
        tab1, tab2, tab3 = st.tabs(["🌡️ Températures", "💧 Précipitations", "🔗 Corrélations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Évolution Temporelle Animée")
                fig_temp = create_temperature_evolution(df)
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                st.markdown("#### Heatmap Interactive")
                st.plotly_chart(
                    create_interactive_heatmap(df),
                    use_container_width=True
                )
        
        with tab2:
            st.markdown("#### Précipitations Animées")
            fig_prcp = create_precipitation_chart(df)
            st.plotly_chart(fig_prcp, use_container_width=True)
        
        with tab3:
            # Matrice de corrélation
            st.markdown("#### Matrice de Corrélation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                if isinstance(df, dd.DataFrame):
                    with ProgressBar():
                        corr_matrix = df[numeric_cols].corr().compute()
                else:
                    corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               color_continuous_scale='RdBu', range_color=[-1, 1])
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🌐 Visualisations 3D":
        st.title("🌐 Visualisations 3D Interactives")
        
        fig_3d = create_3d_scatter_plot(df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    elif page == "🗺️ Carte Animée":
        st.title("🗺️ Carte Climatique Animée")
        
        if 'lat' in df.columns and 'lon' in df.columns:
            fig_map = create_animated_temperature_map(df)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Les données géographiques ne sont pas disponibles")
    
    elif page == "🔬 Avancé":
        st.title("🔬 Analyses Avancées")
        
        # Analyse de tendance
        st.markdown("#### Analyse de Tendance")
        
        if 'tavg' in df.columns and 'year' in df.columns:
            if isinstance(df, dd.DataFrame):
                with ProgressBar():
                    yearly_avg = df.groupby('year')['tavg'].mean().compute().reset_index()
            else:
                yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
            
            if len(yearly_avg) > 1:
                coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                trend_line = np.poly1d(coeffs)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_avg['year'],
                    y=yearly_avg['tavg'],
                    mode='markers',
                    name='Données',
                    marker=dict(size=10)
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_avg['year'],
                    y=trend_line(yearly_avg['year']),
                    mode='lines',
                    name=f'Tendance ({coeffs[0]*10:.3f}°C/décennie)',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title='📈 Analyse de Tendance Linéaire',
                    xaxis_title='Année',
                    yaxis_title='Température Moyenne (°C)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Radar & Parallèles":
        st.title("🎯 Visualisations Avancées")
        st.info("Cette page nécessite des données structurées")
    
    # Footer
    st.markdown("---")
    tech_stack = "Dask + Datashader + HoloViews" if DATA_VIZ_ENABLED else "Pandas + Plotly"
    st.markdown(f"""
    <div style='text-align: center'>
        <p>🌍 AgriClima360 - Dashboard Climatique Avancé</p>
        <p style='font-size: 0.8em; color: gray;'>
            Tech: {tech_stack} | 
            Données: {len(df):,} points | 
            NOAA API
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
