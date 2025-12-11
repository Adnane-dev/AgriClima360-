# =============================================================
# AGRICLIMA360 - Application Streamlit avec données NOAA API
# Visualisations climatiques interactives
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

# Configuration de la page
st.set_page_config(
    page_title="AgriClima360 - Dashboard Climatique",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# 1. CONFIGURATION API NOAA
# =============================================================

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/"

# ⚠️ IMPORTANT: Obtenez votre token gratuit sur https://www.ncdc.noaa.gov/cdo-web/token
NOAA_TOKEN = st.secrets.get("NOAA_TOKEN", "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM")  # À configurer dans .streamlit/secrets.toml

@st.cache_data(ttl=3600)
def get_noaa_data(endpoint, params=None, token=NOAA_TOKEN):
    """
    Récupère les données depuis l'API NOAA.
    
    Args:
        endpoint: Point de terminaison de l'API (datasets, stations, data, etc.)
        params: Paramètres de la requête
        token: Token d'authentification NOAA
    """
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
    """
    Récupère les données climatiques depuis NOAA.
    
    Args:
        dataset_id: ID du dataset (GHCND = Global Historical Climatology Network Daily)
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        location_id: ID de localisation (ex: FIPS:US)
        datatypes: Liste des types de données (TMAX, TMIN, PRCP, etc.)
        limit: Nombre maximum de résultats
    """
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

@st.cache_data(ttl=86400)
def get_available_stations(dataset_id="GHCND", location_id=None, limit=100):
    """Récupère la liste des stations météo disponibles."""
    params = {
        "datasetid": dataset_id,
        "limit": limit
    }
    
    if location_id:
        params["locationid"] = location_id
    
    data = get_noaa_data("stations", params)
    
    if data and "results" in data:
        return pd.DataFrame(data["results"])
    
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_available_locations(limit=100):
    """Récupère la liste des localisations disponibles."""
    params = {"limit": limit}
    data = get_noaa_data("locations", params)
    
    if data and "results" in data:
        return pd.DataFrame(data["results"])
    
    return pd.DataFrame()

# =============================================================
# 2. FONCTIONS DE TRAITEMENT DES DONNÉES
# =============================================================

def process_climate_data(df):
    """Traite et enrichit les données climatiques."""
    if df.empty:
        return generate_sample_data()
    
    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Conversion des températures (de dixièmes de degrés Celsius)
    if 'value' in df.columns:
        # Les températures NOAA sont en dixièmes de degrés
        temp_types = ['TMAX', 'TMIN', 'TAVG']
        df.loc[df['datatype'].isin(temp_types), 'value'] = df.loc[df['datatype'].isin(temp_types), 'value'] / 10
        
        # Les précipitations sont en dixièmes de mm
        df.loc[df['datatype'] == 'PRCP', 'value'] = df.loc[df['datatype'] == 'PRCP', 'value'] / 10
    
    # Pivoter pour avoir les différents types de données en colonnes
    df_pivot = df.pivot_table(
        index=['date', 'year', 'month', 'day', 'station'],
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
        'SNWD': 'snow_depth'
    }
    df_pivot = df_pivot.rename(columns=column_mapping)
    
    # Calculer tavg si manquant
    if 'tavg' not in df_pivot.columns and 'tmax' in df_pivot.columns and 'tmin' in df_pivot.columns:
        df_pivot['tavg'] = (df_pivot['tmax'] + df_pivot['tmin']) / 2
    
    # Ajouter des données simulées pour les visualisations avancées
    df_pivot['humidity'] = np.random.uniform(30, 90, len(df_pivot))
    df_pivot['wind_speed'] = np.random.uniform(0, 20, len(df_pivot))
    df_pivot['continent'] = 'North America'  # À adapter selon la localisation
    df_pivot['lat'] = 40.0 + np.random.uniform(-5, 5, len(df_pivot))
    df_pivot['lon'] = -100.0 + np.random.uniform(-10, 10, len(df_pivot))
    
    return df_pivot

def generate_sample_data():
    """Génère des données de démonstration si l'API n'est pas disponible."""
    st.warning("⚠️ Utilisation de données de démonstration. Configurez votre token NOAA pour des données réelles.")
    
    years = list(range(2020, 2024))
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    data = []
    for date in dates:
        warming_trend = 0.03 * (date.year - 2020)
        base_temp = 15 + warming_trend + 10 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        data.append({
            'date': date,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'station': 'GHCND:FAM00014100',
            'tavg': base_temp + np.random.normal(0, 2),
            'tmax': base_temp + 5 + np.random.normal(0, 2),
            'tmin': base_temp - 5 + np.random.normal(0, 2),
            'prcp': max(0, np.random.exponential(5)),
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 20),
            'continent': 'North America',
            'lat': 40.0 + np.random.uniform(-2, 2),
            'lon': -100.0 + np.random.uniform(-5, 5)
        })
    
    return pd.DataFrame(data)

def compute_kpis(df):
    """Calcule les indicateurs clés."""
    return {
        "temp_moy": df["tavg"].mean() if "tavg" in df.columns else 0,
        "temp_trend": np.polyfit(df['year'].unique(), df.groupby('year')['tavg'].mean().values, 1)[0] * 100 if "tavg" in df.columns else 0,
        "pluie_totale": df["prcp"].sum() if "prcp" in df.columns else 0,
        "nb_annees": df["year"].nunique(),
        "temp_max": df["tmax"].max() if "tmax" in df.columns else 0,
        "temp_min": df["tmin"].min() if "tmin" in df.columns else 0,
        "humidite_moy": df["humidity"].mean() if "humidity" in df.columns else 65,
        "variability": df.groupby('year')['tavg'].std().mean() if "tavg" in df.columns else 0,
        "heatwaves": (df['tmax'] > 30).sum() / len(df) * 100 if "tmax" in df.columns and len(df) > 0 else 0,
        "drought_risk": (df['prcp'] < 5).sum() / len(df) * 100 if "prcp" in df.columns and len(df) > 0 else 0
    }

# =============================================================
# 3. FONCTIONS DE VISUALISATION
# =============================================================

def create_temperature_evolution(df):
    """Crée le graphique d'évolution des températures."""
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
        line=dict(color='red', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tavg'],
        name='Température Moyenne',
        mode='lines+markers',
        line=dict(color='orange', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tmin'],
        name='Température Min',
        mode='lines+markers',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title='📈 Évolution des Températures',
        xaxis_title='Année',
        yaxis_title='Température (°C)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_precipitation_chart(df):
    """Crée le graphique des précipitations."""
    monthly_prcp = df.groupby(['year', 'month'])['prcp'].sum().reset_index()
    
    fig = px.bar(
        monthly_prcp,
        x='month',
        y='prcp',
        color='year',
        title='💧 Précipitations Mensuelles',
        labels={'month': 'Mois', 'prcp': 'Précipitations (mm)', 'year': 'Année'},
        height=500
    )
    
    return fig

def create_heatmap(df):
    """Crée une heatmap des températures."""
    pivot_data = df.pivot_table(
        index='month',
        columns='year',
        values='tavg',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
           'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'],
        colorscale='Viridis',
        colorbar=dict(title="Température (°C)")
    ))
    
    fig.update_layout(
        title='📅 Heatmap des Températures',
        xaxis_title='Année',
        yaxis_title='Mois',
        height=500
    )
    
    return fig

def create_climate_map(df):
    """Crée une carte des données climatiques."""
    map_data = df.groupby(['lat', 'lon', 'year']).agg({
        'tavg': 'mean',
        'prcp': 'sum'
    }).reset_index()
    
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        color='tavg',
        size='prcp',
        animation_frame='year',
        color_continuous_scale='Viridis',
        title='🗺️ Carte Climatique Interactive',
        height=600
    )
    
    fig.update_layout(geo=dict(
        showland=True,
        landcolor="lightgray"
    ))
    
    return fig

def create_3d_visualization(df):
    """Crée une visualisation 3D."""
    sample_df = df.sample(min(500, len(df)))
    
    fig = px.scatter_3d(
        sample_df,
        x='tavg',
        y='prcp',
        z='humidity',
        color='month',
        title='🌐 Visualisation 3D des Variables Climatiques',
        labels={
            'tavg': 'Température (°C)',
            'prcp': 'Précipitations (mm)',
            'humidity': 'Humidité (%)'
        },
        height=600
    )
    
    return fig

def create_correlation_matrix(df):
    """Crée une matrice de corrélation."""
    numeric_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    corr = df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Corrélation")
    ))
    
    fig.update_layout(
        title='🔗 Matrice de Corrélation',
        height=500
    )
    
    return fig

# =============================================================
# 4. INTERFACE STREAMLIT
# =============================================================

def main():
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("🌾 AgriClima360")
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
                    ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW"],
                    default=["TMAX", "TMIN", "PRCP"]
                )
                
                limit = st.slider("Nombre de résultats:", 100, 10000, 1000)
        
        st.markdown("---")
        
        # Navigation
        st.header("📊 Navigation")
        page = st.radio(
            "Sections:",
            ["🏠 Vue d'ensemble", "📈 Analyses", "🗺️ Carte", "🔬 Avancé"]
        )
        
        st.markdown("---")
        
        # Filtres
        st.header("🎛️ Filtres")
        
    # Chargement des données
    with st.spinner("⏳ Chargement des données..."):
        if data_source == "API NOAA (Réelles)":
            if NOAA_TOKEN == "YOUR_TOKEN_HERE":
                st.error("❌ Token NOAA non configuré. Créez un fichier `.streamlit/secrets.toml` avec:\n```toml\nNOAA_TOKEN = 'votre_token'\n```")
                df = generate_sample_data()
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
            df = generate_sample_data()
    
    # Vérification des données
    if df.empty:
        st.error("❌ Aucune donnée disponible. Vérifiez vos paramètres.")
        return
    
    # Calcul des KPIs
    kpis = compute_kpis(df)
    
    # Filtres dans la sidebar
    with st.sidebar:
        if 'year' in df.columns:
            year_range = st.slider(
                "Période:",
                int(df['year'].min()),
                int(df['year'].max()),
                (int(df['year'].min()), int(df['year'].max()))
            )
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # =============================================================
    # PAGES
    # =============================================================
    
    if page == "🏠 Vue d'ensemble":
        st.title("🌍 AgriClima360 - Dashboard Climatique")
        st.markdown("### Visualisations interactives des données climatiques NOAA")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
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
                "📊 Variabilité",
                f"{kpis['variability']:.2f}°C",
                f"Min: {kpis['temp_min']:.1f}°C"
            )
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_temperature_evolution(df),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_precipitation_chart(df),
                use_container_width=True
            )
        
        st.plotly_chart(
            create_heatmap(df),
            use_container_width=True
        )
    
    elif page == "📈 Analyses":
        st.title("📊 Analyses Avancées")
        
        tab1, tab2, tab3 = st.tabs(["📈 Tendances", "🔗 Corrélations", "🌐 3D"])
        
        with tab1:
            st.plotly_chart(
                create_temperature_evolution(df),
                use_container_width=True
            )
            
            st.plotly_chart(
                create_precipitation_chart(df),
                use_container_width=True
            )
        
        with tab2:
            st.plotly_chart(
                create_correlation_matrix(df),
                use_container_width=True
            )
            
            # Statistiques descriptives
            st.subheader("📊 Statistiques Descriptives")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.dataframe(df[numeric_cols].describe())
        
        with tab3:
            st.plotly_chart(
                create_3d_visualization(df),
                use_container_width=True
            )
    
    elif page == "🗺️ Carte":
        st.title("🗺️ Carte Climatique Interactive")
        
        st.plotly_chart(
            create_climate_map(df),
            use_container_width=True
        )
        
        # Statistiques par région
        st.subheader("📍 Statistiques Géographiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🌐 Latitude Moyenne", f"{df['lat'].mean():.2f}°")
            st.metric("🌐 Longitude Moyenne", f"{df['lon'].mean():.2f}°")
        
        with col2:
            st.metric("📏 Étendue Lat.", f"{df['lat'].max() - df['lat'].min():.2f}°")
            st.metric("📏 Étendue Lon.", f"{df['lon'].max() - df['lon'].min():.2f}°")
    
    elif page == "🔬 Avancé":
        st.title("🔬 Analyses Avancées")
        
        # Options de visualisation personnalisées
        st.subheader("🎨 Créer une visualisation personnalisée")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox(
                "Type de graphique:",
                ["Ligne", "Barre", "Scatter", "Box", "Violin"]
            )
        
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            x_var = st.selectbox("Variable X:", numeric_cols)
        
        with col3:
            y_var = st.selectbox("Variable Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        color_var = st.selectbox(
            "Couleur par:",
            [None] + ['year', 'month', 'continent']
        )
        
        # Créer le graphique personnalisé
        if chart_type == "Ligne":
            fig = px.line(df, x=x_var, y=y_var, color=color_var)
        elif chart_type == "Barre":
            fig = px.bar(df, x=x_var, y=y_var, color=color_var)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_var, y=y_var, color=color_var)
        elif chart_type == "Box":
            fig = px.box(df, x=x_var, y=y_var, color=color_var)
        else:  # Violin
            fig = px.violin(df, x=x_var, y=y_var, color=color_var)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export des données
        st.subheader("💾 Export des Données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger CSV",
                csv,
                "climate_data.csv",
                "text/csv"
            )
        
        with col2:
            json_str = df.to_json(orient='records')
            st.download_button(
                "📥 Télécharger JSON",
                json_str,
                "climate_data.json",
                "application/json"
            )
        
        with col3:
            st.info("Excel disponible via pandas.to_excel()")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌍 AgriClima360 - Dashboard Climatique avec données NOAA</p>
            <p style='font-size: 0.8em; color: gray;'>
                Données fournies par NOAA National Centers for Environmental Information
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()