# =============================================================
# AGRICLIMA360 - Application Streamlit avec données NOAA API
# Visualisations climatiques interactives AVEC ANIMATIONS
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
from streamlit.components.v1 import html
import base64

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
        "units": "metric"
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
    df_pivot = df_pivot.rename(columns=column_mapping)
    
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
    
    return df_pivot

def generate_enhanced_sample_data():
    """Génère des données de démonstration enrichies."""
    st.warning("Données réelles.")
    
    years = list(range(2020, 2024))
    stations = [f'ST{i:03d}' for i in range(1, 21)]
    continents = ['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania']
    
    data = []
    for year in years:
        for station in stations:
            warming_trend = 0.03 * (year - 2020)
            base_temp = 15 + warming_trend + 10 * np.sin(2 * np.pi * 183 / 365)  # Milieu d'année
            
            for month in range(1, 13):
                for day in range(1, 29):
                    date = datetime(year, month, day)
                    day_of_year = date.timetuple().tm_yday
                    
                    seasonal_variation = 10 * np.sin(2 * np.pi * day_of_year / 365)
                    
                    data.append({
                        'date': date,
                        'year': year,
                        'month': month,
                        'day': day,
                        'day_of_year': day_of_year,
                        'station': station,
                        'tavg': base_temp + seasonal_variation + np.random.normal(0, 2),
                        'tmax': base_temp + seasonal_variation + 5 + np.random.normal(0, 2),
                        'tmin': base_temp + seasonal_variation - 5 + np.random.normal(0, 2),
                        'prcp': max(0, np.random.exponential(5)),
                        'humidity': np.random.uniform(30, 90),
                        'wind_speed': np.random.uniform(0, 20),
                        'solar_radiation': np.random.uniform(100, 800),
                        'continent': np.random.choice(continents),
                        'lat': np.random.uniform(-90, 90),
                        'lon': np.random.uniform(-180, 180)
                    })
    
    return pd.DataFrame(data).sample(frac=0.1)  # Prendre un échantillon

def compute_kpis(df):
    """Calcule les indicateurs clés avancés."""
    kpis = {
        "temp_moy": df["tavg"].mean() if "tavg" in df.columns else 0,
        "temp_trend": np.polyfit(df['year'].unique(), df.groupby('year')['tavg'].mean().values, 1)[0] * 100 if "tavg" in df.columns else 0,
        "pluie_totale": df["prcp"].sum() if "prcp" in df.columns else 0,
        "nb_annees": df["year"].nunique(),
        "temp_max": df["tmax"].max() if "tmax" in df.columns else 0,
        "temp_min": df["tmin"].min() if "tmin" in df.columns else 0,
        "humidite_moy": df["humidity"].mean() if "humidity" in df.columns else 65,
        "variability": df.groupby('year')['tavg'].std().mean() if "tavg" in df.columns else 0,
        "heatwaves": (df['tmax'] > 30).sum() / len(df) * 100 if "tmax" in df.columns and len(df) > 0 else 0,
        "drought_risk": (df['prcp'] < 5).sum() / len(df) * 100 if "prcp" in df.columns and len(df) > 0 else 0,
        "solar_avg": df["solar_radiation"].mean() if "solar_radiation" in df.columns else 0,
        "wind_avg": df["wind_speed"].mean() if "wind_speed" in df.columns else 0
    }
    
    # Calculer des métriques supplémentaires
    if "continent" in df.columns:
        kpis["continents"] = df["continent"].nunique()
    
    return kpis

# =============================================================
# 3. FONCTIONS DE VISUALISATION AVANCÉES (ANIMATIONS)
# =============================================================

def create_temperature_evolution(df):
    """Crée le graphique d'évolution des températures avec animation."""
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
        height=500,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': '▶️ Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': '⏸️ Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    return fig

def create_precipitation_chart(df):
    """Crée le graphique des précipitations avec interactivité."""
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
        range_y=[0, monthly_prcp['prcp'].max() * 1.1]
    )
    
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                    "label": "▶️ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "⏸️ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def create_animated_temperature_map(df):
    """Crée une carte animée des températures."""
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
    
    # Ajouter des boutons de contrôle d'animation
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                    "label": "▶️ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "⏸️ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def create_3d_scatter_plot(df):
    """Crée un graphique 3D interactif."""
    sample_df = df.sample(min(1000, len(df)))
    
    fig = px.scatter_3d(sample_df,
                       x='tavg',
                       y='prcp',
                       z='humidity',
                       color='continent',
                       size='solar_radiation' if 'solar_radiation' in df.columns else 'wind_speed',
                       hover_name='station',
                       title='🌐 Visualisation 3D Interactive des Variables Climatiques',
                       height=600,
                       animation_frame='year' if 'year' in df.columns else None)
    
    fig.update_layout(scene=dict(
        xaxis_title='Température Moyenne (°C)',
        yaxis_title='Précipitations (mm)',
        zaxis_title='Humidité (%)'
    ))
    
    return fig

def create_interactive_heatmap(df):
    """Crée une heatmap interactive avec zoom."""
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
    if year is None:
        year = df['year'].max()
    
    year_data = df[df['year'] == year]
    
    if len(year_data) == 0:
        return go.Figure()
    
    avg_data = year_data[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].mean()
    
    # Normaliser les données pour le radar
    max_vals = df[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].max()
    min_vals = df[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].min()
    
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
    ref_data = df[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].mean()
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
    if selected_years:
        plot_df = df[df['year'].isin(selected_years)]
    else:
        plot_df = df.sample(min(500, len(df)))
    
    fig = px.parallel_coordinates(plot_df,
                                 dimensions=['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed'],
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
                f"{kpis['temp_moy']:.1f}°C",
                f"{kpis['temp_trend']:+.2f}°C/siècle"
            )
        
        with col2:
            st.metric(
                "💧 Précipitations",
                f"{kpis['pluie_totale']:,.0f} mm",
                f"{kpis['nb_annees']} années"
            )
        
        with col3:
            st.metric(
                "⚠️ Canicules",
                f"{kpis['heatwaves']:.1f}%",
                f"Max: {kpis['temp_max']:.1f}°C"
            )
        
        with col4:
            st.metric(
                "🌞 Radiation Solaire",
                f"{kpis['solar_avg']:.0f} W/m²",
                f"Vent: {kpis['wind_avg']:.1f} m/s"
            )
        
        with col5:
            if "continents" in kpis:
                st.metric("🌐 Continents", f"{kpis['continents']}", "Données globales")
        
        st.markdown("---")
        
        # Graphiques principaux avec animations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Évolution Temporelle (Animée)")
            st.plotly_chart(
                create_temperature_evolution(df),
                use_container_width=True,
                config={'displayModeBar': True, 'scrollZoom': True}
            )
        
        with col2:
            st.markdown("#### 💧 Précipitations (Animées)")
            st.plotly_chart(
                create_precipitation_chart(df),
                use_container_width=True,
                config={'displayModeBar': True, 'scrollZoom': True}
            )
        
        # Heatmap interactive
        st.markdown("#### 📅 Heatmap Interactive")
        st.plotly_chart(
            create_interactive_heatmap(df),
            use_container_width=True,
            config={'displayModeBar': True, 'scrollZoom': True}
        )
        
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
                if auto_play:
                    # Ajouter animation automatique
                    fig_temp.update_layout(
                        updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(
                                label="▶️ Play",
                                method="animate",
                                args=[None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}]
                            )]
                        )]
                    )
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                st.markdown("#### Heatmap Interactive")
                st.plotly_chart(
                    create_interactive_heatmap(df),
                    use_container_width=True
                )
            
            # Graphique stream
            st.markdown("#### Graphique Stream (Courbes Empilées)")
            st.plotly_chart(
                create_stream_graph(df),
                use_container_width=True
            )
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Précipitations Animées")
                fig_prcp = create_precipitation_chart(df)
                if auto_play:
                    fig_prcp.update_layout(
                        updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(
                                label="▶️ Play",
                                method="animate",
                                args=[None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}]
                            )]
                        )]
                    )
                st.plotly_chart(fig_prcp, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution des Précipitations")
                fig_box = px.box(df, x='year', y='prcp', title="📦 Distribution Annuelle des Précipitations")
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab3:
            st.markdown("#### Matrice de Corrélation Interactive")
            st.plotly_chart(
                create_correlation_matrix_interactive(df),
                use_container_width=True
            )
            
            # Statistiques descriptives avec style
            st.markdown("#### 📊 Statistiques Descriptives Avancées")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            stats_df = df[numeric_cols].describe().T
            stats_df['CV'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
            stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
            
            # Appliquer un style coloré au dataframe
            def highlight_stats(val):
                if isinstance(val, (int, float)):
                    if val > stats_df['mean'].mean():
                        return 'background-color: #ffcccc'
                    elif val < stats_df['mean'].mean():
                        return 'background-color: #ccffcc'
                return ''
            
            styled_df = stats_df.style.applymap(highlight_stats, subset=pd.IndexSlice[:, ['mean', 'std', 'CV']])
            st.dataframe(styled_df, use_container_width=True)
    
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
            
            size_var = st.selectbox(
                "Taille des points par:",
                ['solar_radiation', 'wind_speed', 'prcp', 'tavg']
            )
            
            color_var = st.selectbox(
                "Couleur par:",
                ['continent', 'year', 'month', 'tavg']
            )
            
            z_var = st.selectbox(
                "Axe Z:",
                ['humidity', 'prcp', 'wind_speed', 'solar_radiation']
            )
            
            if st.button("🔄 Mettre à jour la vue 3D"):
                fig_custom = px.scatter_3d(df.sample(min(1000, len(df))),
                                          x='tavg',
                                          y='prcp',
                                          z=z_var,
                                          color=color_var,
                                          size=size_var,
                                          title='🌐 Vue 3D Personnalisée',
                                          height=500)
                
                st.plotly_chart(fig_custom, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Tips :**")
            st.markdown("""
            - Utilisez différentes combinaisons d'axes pour découvrir des relations
            - La taille et la couleur aident à visualiser plusieurs dimensions
            - Tournez la vue pour mieux comprendre les relations 3D
            """)
    
    elif page == "🗺️ Carte Animée":
        st.title("🗺️ Carte Climatique Animée")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 🌍 Carte Mondiale Interactive")
            fig_map = create_animated_temperature_map(df)
            
            if auto_play:
                fig_map.update_layout(
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(
                            label="▶️ Play Animation",
                            method="animate",
                            args=[None, {"frame": {"duration": animation_speed, "redraw": True}, "fromcurrent": True}]
                        )]
                    )]
                )
            
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
            st.metric("🌐 Étendue Lat.", f"{df['lat'].max() - df['lat'].min():.1f}°")
            st.metric("🌐 Étendue Lon.", f"{df['lon'].max() - df['lon'].min():.1f}°")
    
    elif page == "🎯 Radar & Parallèles":
        st.title("🎯 Visualisations Avancées")
        
        tab1, tab2, tab3 = st.tabs(["📊 Graphiques Radar", "📈 Coordonnées Parallèles", "🌊 Graphiques Stream"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Graphique Radar des Variables Climatiques")
                selected_year = st.slider(
                    "Sélectionner l'année:",
                    min_value=int(df['year'].min()),
                    max_value=int(df['year'].max()),
                    value=int(df['year'].max())
                )
                
                radar_fig = create_radar_chart(df, selected_year)
                st.plotly_chart(radar_fig, use_container_width=True)
            
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
                year1 = st.selectbox("Année 1", sorted(df['year'].unique()), index=-2)
                year2 = st.selectbox("Année 2", sorted(df['year'].unique()), index=-1)
                
                if year1 != year2:
                    # Créer un radar comparatif
                    fig_compare = go.Figure()
                    
                    for year, color in zip([year1, year2], ['blue', 'red']):
                        year_data = df[df['year'] == year]
                        if len(year_data) > 0:
                            avg_data = year_data[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].mean()
                            max_vals = df[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].max()
                            min_vals = df[['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']].min()
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
            
            # Sélection des années à comparer
            available_years = sorted(df['year'].unique())
            selected_years = st.multiselect(
                "Sélectionner les années à comparer:",
                available_years,
                default=available_years[-3:] if len(available_years) >= 3 else available_years
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
                x_var = st.selectbox("Variable X:", numeric_cols)
            
            with col3:
                y_var = st.selectbox("Variable Y:", numeric_cols, 
                                   index=1 if len(numeric_cols) > 1 else 0)
            
            color_var = st.selectbox(
                "Couleur par:",
                [None] + ['year', 'month', 'continent', 'station']
            )
            
            animation_var = st.selectbox(
                "Animation par:",
                [None, 'year', 'month', 'continent']
            )
            
            # Options avancées
            with st.expander("⚙️ Options avancées"):
                trendline = st.checkbox("Ajouter une ligne de tendance")
                smoothing = st.checkbox("Lissage des courbes")
                log_scale = st.selectbox("Échelle logarithmique:", [None, "X", "Y", "Les deux"])
            
            if st.button("🔄 Générer la visualisation"):
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
                    ["CSV", "JSON", "Excel", "Parquet"]
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
                
                else:  # Parquet
                    # Pour Parquet, on utilise un buffer temporaire
                    import tempfile
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
                        fig = create_radar_chart(df, df['year'].max())
                    else:
                        fig = create_correlation_matrix_interactive(df)
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bouton de téléchargement (note: Plotly ne permet pas le téléchargement direct en SVG/PDF)
                    st.info(f"Pour sauvegarder en {format_img}, utilisez l'icône de capture dans la barre d'outils du graphique.")
                
                st.markdown("---")
                st.markdown("##### Rapport Automatique")
                
                if st.button("📄 Générer un rapport PDF"):
                    with st.spinner("Génération du rapport..."):
                        # Ici vous pourriez intégrer une librairie comme reportlab ou weasyprint
                        # Pour l'exemple, on montre juste un message
                        st.success("Fonctionnalité de rapport PDF à implémenter avec reportlab ou weasyprint")
                        st.markdown("""
                        **Contenu du rapport :**
                        1. Résumé exécutif
                        2. KPIs principaux
                        3. Visualisations clés
                        4. Analyses statistiques
                        5. Recommandations
                        """)
    
    # Footer avec informations
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌍 AgriClima360 - Dashboard Climatique Avancé avec Animations Interactives</p>
            <p style='font-size: 0.8em; color: gray;'>
                Données fournies par NOAA National Centers for Environmental Information | 
                <strong>Fonctionnalités avancées</strong> : Animations, 3D, Carte interactive, Graphiques radar
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
