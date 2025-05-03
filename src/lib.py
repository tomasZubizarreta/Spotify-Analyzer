import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SpotifyAnalyzer:
    def __init__(self, data_file=None, data_dir="spotify_data"):
        """
        Inicializa el analizador de datos de Spotify.
        
        Args:
            data_file: Ruta al archivo principal de datos (si se proporciona)
            data_dir: Ruta al directorio con los datos de Spotify
        """
        self.data_dir = data_dir
        
        # Archivos de datos
        self.track_history_file = data_file if data_file else os.path.join(data_dir, "track_history.json")
        self.combined_history_file = os.path.join(data_dir, "all_tracks_combined.json")
        self.recent_tracks_file = os.path.join(data_dir, "recently_played.json")
        
        # DataFrames para diferentes tipos de datos
        self.current_df = None
        self.recent_df = None
        self.combined_df = None
        self.top_tracks = {
            'short_term': None,
            'medium_term': None,
            'long_term': None
        }
        self.top_artists = {
            'short_term': None,
            'medium_term': None,
            'long_term': None
        }
        
        # Cargar todos los datos disponibles
        self.load_all_data()
        
        # Directorio para guardar las visualizaciones
        self.output_dir = os.path.join(data_dir, "analysis")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Paleta de colores para visualizaciones
        self.colors = {
            'primary': '#1DB954',  # Verde Spotify
            'secondary': '#191414',  # Negro Spotify
            'accent': '#FFFFFF',    # Blanco
            'palette': ['#1DB954', '#1ED760', '#2D46B9', '#F1C40F', '#E74C3C', '#9B59B6', '#3498DB']
        }
    
    def load_all_data(self):
        """Carga todos los archivos de datos disponibles."""
        # Cargar datos de seguimiento actual (track_history.json)
        self._load_current_data()
        
        # Cargar datos combinados si existen
        self._load_combined_data()
        
        # Cargar datos recientes si existen
        self._load_recent_data()
        
        # Cargar archivos individuales de top tracks/artists si existen
        self._load_top_data()
    
    def _load_current_data(self):
        """Carga datos de seguimiento actual"""
        if os.path.exists(self.track_history_file):
            try:
                with open(self.track_history_file, 'r', encoding='utf-8') as f:
                    tracks = json.load(f)
                self.current_df = pd.DataFrame(tracks)
                
                # Convertir timestamp a datetime
                if 'timestamp' in self.current_df.columns:
                    self.current_df['datetime'] = pd.to_datetime(self.current_df['timestamp'])
                    # Añadir columnas de fecha y hora
                    self.current_df['date'] = self.current_df['datetime'].dt.date
                    self.current_df['hour'] = self.current_df['datetime'].dt.hour
                    self.current_df['day_of_week'] = self.current_df['datetime'].dt.day_name()
                
                print(f"Datos de seguimiento actual cargados: {len(self.current_df)} canciones")
            except Exception as e:
                print(f"Error al cargar {self.track_history_file}: {e}")
    
    def _load_combined_data(self):
        """Carga datos combinados"""
        if os.path.exists(self.combined_history_file):
            try:
                with open(self.combined_history_file, 'r', encoding='utf-8') as f:
                    tracks = json.load(f)
                self.combined_df = pd.DataFrame(tracks)
                
                # Convertir timestamp a datetime
                timestamp_cols = [col for col in ['timestamp', 'played_at', 'added_at'] 
                                 if col in self.combined_df.columns]
                
                if timestamp_cols:
                    # Usar la primera columna de timestamp disponible
                    self.combined_df['datetime'] = pd.to_datetime(self.combined_df[timestamp_cols[0]])
                    # Añadir columnas de fecha y hora
                    self.combined_df['date'] = self.combined_df['datetime'].dt.date
                    self.combined_df['hour'] = self.combined_df['datetime'].dt.hour
                    self.combined_df['day_of_week'] = self.combined_df['datetime'].dt.day_name()
                
                print(f"Datos combinados cargados: {len(self.combined_df)} canciones")
            except Exception as e:
                print(f"Error al cargar {self.combined_history_file}: {e}")
    
    def _load_recent_data(self):
        """Carga datos de reproducción reciente"""
        if os.path.exists(self.recent_tracks_file):
            try:
                with open(self.recent_tracks_file, 'r', encoding='utf-8') as f:
                    recent_tracks = json.load(f)
                self.recent_df = pd.DataFrame(recent_tracks)
                
                # Convertir played_at a datetime
                if 'played_at' in self.recent_df.columns:
                    self.recent_df['datetime'] = pd.to_datetime(self.recent_df['played_at'])
                    # Añadir columnas de fecha y hora
                    self.recent_df['date'] = self.recent_df['datetime'].dt.date
                    self.recent_df['hour'] = self.recent_df['datetime'].dt.hour
                    self.recent_df['day_of_week'] = self.recent_df['datetime'].dt.day_name()
                
                print(f"Datos de reproducción reciente cargados: {len(self.recent_df)} canciones")
            except Exception as e:
                print(f"Error al cargar {self.recent_tracks_file}: {e}")
    
    def _load_top_data(self):
        """Carga datos de top tracks y artistas"""
        for time_range, desc in [('short_term', 'ultimo_mes'), 
                                ('medium_term', 'ultimos_6_meses'), 
                                ('long_term', 'todo_el_tiempo')]:
            track_file = os.path.join(self.data_dir, f"top_tracks_{desc}.json")
            if os.path.exists(track_file):
                try:
                    with open(track_file, 'r', encoding='utf-8') as f:
                        tracks = json.load(f)
                    self.top_tracks[time_range] = pd.DataFrame(tracks)
                    print(f"Top tracks ({desc}) cargados: {len(self.top_tracks[time_range])} canciones")
                except Exception as e:
                    print(f"Error al cargar {track_file}: {e}")
            
            artist_file = os.path.join(self.data_dir, f"top_artists_{desc}.json")
            if os.path.exists(artist_file):
                try:
                    with open(artist_file, 'r', encoding='utf-8') as f:
                        artists = json.load(f)
                    self.top_artists[time_range] = pd.DataFrame(artists)
                    print(f"Top artistas ({desc}) cargados: {len(self.top_artists[time_range])} artistas")
                except Exception as e:
                    print(f"Error al cargar {artist_file}: {e}")
    
    def get_listening_patterns(self):
        """
        Analiza patrones de escucha por día de la semana y hora del día.
        Devuelve la ruta al archivo HTML generado.
        """
        # Usar el DataFrame más completo disponible
        df = self._get_best_dataframe()
        
        if df is None or df.empty or 'datetime' not in df.columns:
            print("No hay datos suficientes para analizar patrones de escucha")
            return None
        
        # Asegurarse de que tenemos las columnas necesarias
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.day_name()
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
        # Patrones por día de la semana
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        days_mapping = dict(zip(days_order, days_es))
        
        day_counts = df['day_of_week'].value_counts().reindex(days_order)
        day_counts.index = day_counts.index.map(days_mapping)
        
        # Patrones por hora del día
        hour_counts = df['hour'].value_counts().sort_index()
        
        # Heatmap combinado día de la semana x hora
        day_hour_df = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        day_hour_df = day_hour_df.reindex(days_order)
        day_hour_df.index = day_hour_df.index.map(days_mapping)
        
        # Crear gráfico interactivo con Plotly
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=(
                'Canciones por día de la semana', 
                'Canciones por hora del día',
                'Heatmap: Día de la semana x Hora'
            ),
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "heatmap"}]]
        )
        
        # Gráfico por día
        fig.add_trace(
            go.Bar(
                x=day_counts.index, 
                y=day_counts.values, 
                name='Días',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # Gráfico por hora
        fig.add_trace(
            go.Bar(
                x=hour_counts.index, 
                y=hour_counts.values, 
                name='Horas',
                marker_color=self.colors['primary']
            ),
            row=2, col=1
        )
        
        # Heatmap día x hora
        fig.add_trace(
            go.Heatmap(
                z=day_hour_df.values,
                x=[f"{h}:00" for h in day_hour_df.columns],
                y=day_hour_df.index,
                colorscale='Viridis',
                name='Heatmap'
            ),
            row=3, col=1
        )
        
        # Configuración del layout
        fig.update_layout(
            height=1000,
            title_text="Patrones de escucha",
            showlegend=False,
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar gráfico
        output_file = os.path.join(self.output_dir, "listening_patterns.html")
        fig.write_html(output_file)
        
        print(f"Visualización de patrones de escucha guardada en: {output_file}")
        return output_file
    
    def analyze_mood_patterns(self):
        """
        Analiza patrones de estado de ánimo basados en atributos de audio.
        Devuelve la ruta al archivo HTML generado.
        """
        # Usar el DataFrame combinado o el que tenga atributos de audio
        df = self._get_best_dataframe_for_audio_features()
        
        if df is None or df.empty:
            print("No hay datos suficientes con atributos de audio para analizar")
            return None
        
        # Comprobar si tenemos atributos de audio
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Contar cuántos atributos tenemos disponibles
        available_features = [feat for feat in audio_features if feat in df.columns]
        
        if len(available_features) < 3:
            print(f"No hay suficientes atributos de audio ({len(available_features)}/9)")
            
            # Si no hay atributos de audio, crear una visualización alternativa
            # basada en la popularidad y duración de las canciones
            if 'popularity' in df.columns and 'duration_ms' in df.columns:
                return self._create_popularity_duration_viz(df)
            return None
        
        # Preparar datos para visualización
        mood_df = df[available_features + ['name', 'artist']].copy()
        mood_df['track'] = mood_df['name'] + ' - ' + mood_df['artist']
        
        # Normalizar valores para radar chart (entre 0 y 1)
        for feature in available_features:
            if feature == 'loudness':  # Loudness suele estar en decibelios (-60 a 0)
                mood_df[feature] = (mood_df[feature] + 60) / 60
            elif feature == 'tempo':  # Tempo suele estar entre 50 y 200 BPM
                mood_df[feature] = (mood_df[feature] - 50) / 150
            # El resto de features ya están entre 0 y 1
        
        # Calcular promedios para cada característica
        avg_features = mood_df[available_features].mean().to_dict()
        
        # Crear radar chart
        fig = go.Figure()
        
        # Añadir gráfico de radar
        fig.add_trace(go.Scatterpolar(
            r=list(avg_features.values()),
            theta=list(avg_features.keys()),
            fill='toself',
            name='Perfil de audio promedio',
            line_color=self.colors['primary']
        ))
        
        # Añadir configuración
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Perfil de audio promedio de tu música",
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Crear gráfico scatter para Energy vs Valence (considerado el diagrama emocional)
        if 'energy' in available_features and 'valence' in available_features:
            fig2 = px.scatter(
                mood_df,
                x='valence',
                y='energy',
                hover_name='track',
                title='Mapa emocional de tu música',
                labels={
                    'valence': 'Valencia (tristeza-felicidad)',
                    'energy': 'Energía'
                },
                color_discrete_sequence=[self.colors['primary']]
            )
            
            # Añadir cuadrantes con anotaciones
            fig2.add_shape(
                type="line", x0=0.5, y0=0, x1=0.5, y1=1, 
                line=dict(color="White", width=1, dash="dash")
            )
            fig2.add_shape(
                type="line", x0=0, y0=0.5, x1=1, y1=0.5, 
                line=dict(color="White", width=1, dash="dash")
            )
            
            # Añadir etiquetas para los cuadrantes
            fig2.add_annotation(x=0.25, y=0.75, text="Energético y triste",
                                showarrow=False, font=dict(color="white"))
            fig2.add_annotation(x=0.75, y=0.75, text="Energético y feliz",
                                showarrow=False, font=dict(color="white"))
            fig2.add_annotation(x=0.25, y=0.25, text="Calmado y triste",
                                showarrow=False, font=dict(color="white"))
            fig2.add_annotation(x=0.75, y=0.25, text="Calmado y feliz",
                                showarrow=False, font=dict(color="white"))
            
            fig2.update_layout(
                template="plotly_dark",
                plot_bgcolor=self.colors['secondary'],
                paper_bgcolor=self.colors['secondary'],
                font=dict(color=self.colors['accent'])
            )
            
            # Guardar el segundo gráfico
            fig2_output = os.path.join(self.output_dir, "mood_map.html")
            fig2.write_html(fig2_output)
        
        # Guardar el radar chart
        output_file = os.path.join(self.output_dir, "audio_profile.html")
        fig.write_html(output_file)
        
        print(f"Visualización de perfil de audio guardada en: {output_file}")
        return output_file
    
    def _create_popularity_duration_viz(self, df):
        """Crea visualización alternativa basada en popularidad y duración"""
        if 'popularity' not in df.columns or 'duration_ms' not in df.columns:
            return None
            
        viz_df = df[['name', 'artist', 'popularity', 'duration_ms']].copy()
        viz_df['track'] = viz_df['name'] + ' - ' + viz_df['artist']
        viz_df['duration_min'] = viz_df['duration_ms'] / 60000  # Convertir a minutos
        
        # Crear gráfico scatter
        fig = px.scatter(
            viz_df,
            x='popularity',
            y='duration_min',
            hover_name='track',
            title='Popularidad vs Duración de canciones',
            labels={
                'popularity': 'Popularidad',
                'duration_min': 'Duración (minutos)'
            },
            color_discrete_sequence=[self.colors['primary']]
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar el gráfico
        output_file = os.path.join(self.output_dir, "popularity_duration.html")
        fig.write_html(output_file)
        
        print(f"Visualización alternativa guardada en: {output_file}")
        return output_file
    
    def cluster_music_taste(self, n_clusters=4):
        """
        Agrupa canciones por características similares usando clustering.
        Devuelve la ruta al archivo HTML generado.
        """
        # Usar el DataFrame con más atributos de audio
        df = self._get_best_dataframe_for_audio_features()
        
        if df is None or df.empty:
            print("No hay datos suficientes para clustering")
            return None
        
        # Comprobar si tenemos atributos de audio
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Contar cuántos atributos tenemos disponibles
        available_features = [feat for feat in audio_features if feat in df.columns]
        
        if len(available_features) < 3:
            print(f"No hay suficientes atributos de audio para clustering ({len(available_features)}/9)")
            return None
        
        # Preparar datos para clustering
        cluster_df = df[available_features].copy()
        
        # Normalizar datos (importante para clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df[available_features])
        
        # Aplicar K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Añadir etiquetas de cluster al DataFrame
        df['cluster'] = cluster_labels
        
        # Reducir dimensionalidad para visualización (PCA a 2D)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Añadir resultados PCA al DataFrame
        df['pca_x'] = pca_result[:, 0]
        df['pca_y'] = pca_result[:, 1]
        
        # Añadir información adicional para hover
        df['info'] = df['name'] + ' - ' + df['artist']
        
        # Crear visualización de clusters
        fig = px.scatter(
            df,
            x='pca_x',
            y='pca_y',
            color='cluster',
            hover_name='info',
            color_discrete_sequence=self.colors['palette'],
            title='Agrupación de tu música por características de audio'
        )
        
        # Añadir centroides
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        fig.add_trace(
            go.Scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode='markers',
                marker=dict(
                    color='white',
                    size=10,
                    symbol='x'
                ),
                name='Centroides'
            )
        )
        
        # Calcular características promedio por cluster
        cluster_profiles = df.groupby('cluster')[available_features].mean()
        
        # Crear tabla de perfiles de cluster
        cluster_names = [f"Grupo {i+1}" for i in range(n_clusters)]
        
        # Crear gráfico de radar para cada cluster
        fig2 = make_subplots(
            rows=1, cols=n_clusters,
            subplot_titles=cluster_names,
            specs=[[{"type": "polar"} for _ in range(n_clusters)]]
        )
        
        for i, cluster_id in enumerate(cluster_profiles.index):
            fig2.add_trace(
                go.Scatterpolar(
                    r=cluster_profiles.iloc[i].values,
                    theta=available_features,
                    fill='toself',
                    name=f'Grupo {cluster_id+1}',
                    line_color=self.colors['palette'][i % len(self.colors['palette'])]
                ),
                row=1, col=i+1
            )
        
        fig2.update_layout(
            height=500,
            title_text="Perfiles de audio por grupo",
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Obtener top canciones por cluster
        top_tracks_by_cluster = {}
        for cluster_id in range(n_clusters):
            cluster_tracks = df[df['cluster'] == cluster_id].sort_values('popularity', ascending=False).head(10)
            top_tracks_by_cluster[cluster_id] = cluster_tracks[['name', 'artist', 'popularity']].to_dict('records')
        
        # Guardar perfiles de cluster a JSON
        cluster_info = {
            'profiles': cluster_profiles.to_dict('index'),
            'top_tracks': top_tracks_by_cluster
        }
        
        cluster_info_file = os.path.join(self.output_dir, "cluster_info.json")
        with open(cluster_info_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_info, f, ensure_ascii=False, indent=2)
        
        # Actualizar layout del gráfico principal
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar gráficos
        output_file = os.path.join(self.output_dir, "music_clusters.html")
        fig.write_html(output_file)
        
        output_file2 = os.path.join(self.output_dir, "cluster_profiles.html")
        fig2.write_html(output_file2)
        
        print(f"Visualización de clusters guardada en: {output_file}")
        print(f"Perfiles de cluster guardados en: {output_file2}")
        return output_file
    
    def create_dashboard(self):
        """
        Crea un dashboard completo con todas las visualizaciones.
        Devuelve la ruta al archivo HTML generado.
        """
        # Obtener estadísticas básicas
        stats = self.get_basic_stats()
        
        # Generar todas las visualizaciones
        listening_patterns_file = self.get_listening_patterns()
        mood_file = self.analyze_mood_patterns()
        clusters_file = self.cluster_music_taste()
        
        # Visualizar top artistas y canciones
        top_artists_viz = self.visualize_top_artists()
        top_tracks_viz = self.visualize_top_tracks()
        
        # Crear HTML del dashboard
        dashboard_file = os.path.join(self.output_dir, "spotify_dashboard.html")
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Spotify Dashboard</title>
                <style>
                    body {
                        font-family: 'Helvetica Neue', Arial, sans-serif;
                        background-color: #191414;
                        color: white;
                        margin: 0;
                        padding: 0;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    header {
                        background-color: #1DB954;
                        padding: 20px;
                        text-align: center;
                        margin-bottom: 30px;
                    }
                    h1, h2, h3 {
                        margin-top: 0;
                    }
                    .card {
                        background-color: #282828;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                    }
                    .stat-card {
                        background-color: #181818;
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                    }
                    .stat-value {
                        font-size: 28px;
                        font-weight: bold;
                        color: #1DB954;
                        margin: 10px 0;
                    }
                    .viz-container {
                        margin-top: 30px;
                    }
                    .iframe-container {
                        position: relative;
                        overflow: hidden;
                        padding-top: 56.25%; /* 16:9 Aspect Ratio */
                    }
                    .iframe-container iframe {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        border: none;
                    }
                    .viz-tabs {
                        display: flex;
                        margin-bottom: 20px;
                        overflow-x: auto;
                    }
                    .viz-tab {
                        padding: 10px 20px;
                        background-color: #333;
                        border: none;
                        color: white;
                        cursor: pointer;
                        margin-right: 5px;
                        border-radius: 5px 5px 0 0;
                    }
                    .viz-tab.active {
                        background-color: #1DB954;
                    }
                    .viz-content {
                        display: none;
                    }
                    .viz-content.active {
                        display: block;
                    }
                    footer {
                        text-align: center;
                        margin-top: 50px;
                        padding: 20px;
                        color: #aaa;
                    }
                </style>
            </head>
            <body>
                <header>
                    <h1>Tu Análisis de Spotify</h1>
                    <p>Un vistazo completo a tus hábitos de escucha</p>
                </header>
                
                <div class="container">
                    <div class="card">
                        <h2>Resumen de tu actividad</h2>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <p>Canciones analizadas</p>
                                <div class="stat-value">{stats['current_data']['tracks_count']}</div>
                            </div>
                            <div class="stat-card">
                                <p>Canciones únicas</p>
                                <div class="stat-value">{stats['current_data']['unique_tracks_count']}</div>
                            </div>
                            <div class="stat-card">
                                <p>Artistas únicos</p>
                                <div class="stat-value">{stats['current_data']['unique_artists_count']}</div>
                            </div>
                            <div class="stat-card">
                                <p>Horas de escucha</p>
                                <div class="stat-value">{stats['current_data']['total_listening_time_hours']}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="viz-container card">
                        <h2>Visualizaciones</h2>
                        <div class="viz-tabs">
                            <button class="viz-tab active" onclick="openViz(event, 'patterns')">Patrones de escucha</button>
                            <button class="viz-tab" onclick="openViz(event, 'artists')">Top Artistas</button>
                            <button class="viz-tab" onclick="openViz(event, 'tracks')">Top Canciones</button>
                            <button class="viz-tab" onclick="openViz(event, 'mood')">Análisis de Estado de Ánimo</button>
                            <button class="viz-tab" onclick="openViz(event, 'clusters')">Agrupación Musical</button>
                        </div>
                        
                        <div id="patterns" class="viz-content active">
                            <div class="iframe-container">
                                <iframe src="{listening_patterns_file.replace(self.output_dir + '/', '')}" frameborder="0"></iframe>
                            </div>
                        </div>
                        
                        <div id="artists" class="viz-content">
                            <div class="iframe-container">
                                <iframe src="{top_artists_viz['chart_path'].replace(self.output_dir + '/', '') if top_artists_viz else ''}" frameborder="0"></iframe>
                            </div>
                        </div>
                        
                        <div id="tracks" class="viz-content">
                            <div class="iframe-container">
                                <iframe src="{top_tracks_viz['chart_path'].replace(self.output_dir + '/', '') if top_tracks_viz else ''}" frameborder="0"></iframe>
                            </div>
                        </div>
                        
                        <div id="mood" class="viz-content">
                            <div class="iframe-container">
                                <iframe src="{mood_file.replace(self.output_dir + '/', '') if mood_file else ''}" frameborder="0"></iframe>
                            </div>
                        </div>
                        
                        <div id="clusters" class="viz-content">
                            <div class="iframe-container">
                                <iframe src="{clusters_file.replace(self.output_dir + '/', '') if clusters_file else ''}" frameborder="0"></iframe>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Tus Conclusiones</h2>
                        <p>Basado en tu historial de escucha, hemos identificado los siguientes patrones:</p>
                        <ul>
                            <li>Tu día de la semana favorito para escuchar música es <strong>{self._get_favorite_day()}</strong>.</li>
                            <li>Tiendes a escuchar más música alrededor de las <strong>{self._get_favorite_hour()}</strong>.</li>
                            <li>Tus artistas favoritos incluyen a <strong>{self._get_top_artists_names()}</strong>.</li>
                            <li>Tu gusto musical tiende hacia canciones con <strong>{self._get_mood_description()}</strong>.</li>
                        </ul>
                    </div>
                </div>
                
                <footer>
                    <p>Análisis generado con Spotify Manager | {datetime.now().strftime('%Y-%m-%d')}</p>
                </footer>
                
                <script>
                    function openViz(evt, vizName) {
                        var i, vizContent, vizTabs;
                        
                        // Ocultar todas las visualizaciones
                        vizContent = document.getElementsByClassName("viz-content");
                        for (i = 0; i < vizContent.length; i++) {
                            vizContent[i].className = vizContent[i].className.replace(" active", "");
                        }
                        
                        // Desactivar todos los tabs
                        vizTabs = document.getElementsByClassName("viz-tab");
                        for (i = 0; i < vizTabs.length; i++) {
                            vizTabs[i].className = vizTabs[i].className.replace(" active", "");
                        }
                        
                        // Mostrar la visualización actual y activar el tab
                        document.getElementById(vizName).className += " active";
                        evt.currentTarget.className += " active";
                    }
                </script>
            </body>
            </html>
            '''.format(
                stats=stats,
                listening_patterns_file=listening_patterns_file if listening_patterns_file else '',
                mood_file=mood_file if mood_file else '',
                clusters_file=clusters_file if clusters_file else '',
                top_artists_viz=top_artists_viz if top_artists_viz else {'chart_path': ''},
                top_tracks_viz=top_tracks_viz if top_tracks_viz else {'chart_path': ''},
                datetime=datetime
            ))
        
        print(f"Dashboard creado en: {dashboard_file}")
        return dashboard_file
    
    def get_basic_stats(self):
        """Obtiene estadísticas básicas del historial de escucha."""
        stats = {
            'current_data': {
                'tracks_count': 0,
                'unique_tracks_count': 0,
                'unique_artists_count': 0,
                'total_listening_time_hours': 0,
                'first_record': None,
                'last_record': None
            },
            'recent_data': {
                'tracks_count': 0,
                'unique_tracks_count': 0,
                'unique_artists_count': 0,
                'first_record': None,
                'last_record': None
            }
        }
        
        # Usar el DataFrame más completo
        df = self._get_best_dataframe()
        
        if df is not None and not df.empty:
            stats['current_data']['tracks_count'] = len(df)
            
            if 'id' in df.columns:
                stats['current_data']['unique_tracks_count'] = df['id'].nunique()
            
            if 'artist' in df.columns:
                stats['current_data']['unique_artists_count'] = df['artist'].nunique()
            
            if 'duration_ms' in df.columns:
                stats['current_data']['total_listening_time_hours'] = round(df['duration_ms'].sum() / (1000 * 60 * 60), 2)
            
            if 'datetime' in df.columns:
                stats['current_data']['first_record'] = df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')
                stats['current_data']['last_record'] = df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
        
        return stats
    
    def visualize_top_artists(self, time_range='medium_term'):
        """Visualiza los artistas más escuchados para un rango de tiempo específico."""
        # Determinar qué DataFrame usar
        if (self.top_artists[time_range] is not None and 
            not self.top_artists[time_range].empty and 
            'name' in self.top_artists[time_range].columns):
            
            # Usar datos de top_artists
            df = self.top_artists[time_range]
            if 'rank' in df.columns:
                df = df.sort_values('rank')
            top_artists = df['name'].head(15).value_counts()
            
            range_name = {
                'short_term': 'último mes',
                'medium_term': 'últimos 6 meses',
                'long_term': 'de todos los tiempos'
            }
            
            title = f"Artistas más escuchados ({range_name.get(time_range, time_range)})"
            data_source = f"top_artists_{time_range}"
        else:
            # Usar el DataFrame más completo disponible
            df = self._get_best_dataframe()
            
            if df is None or df.empty or 'artist' not in df.columns:
                return None
                
            top_artists = df['artist'].value_counts().head(15)
            title = "Artistas más escuchados"
            data_source = "combined_data"
        
        # Crear gráfico con Plotly
        fig = px.bar(
            x=top_artists.values,
            y=top_artists.index,
            orientation='h',
            labels={'x': 'Frecuencia', 'y': 'Artista'},
            title=title,
            color=top_artists.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            width=800,
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar gráfico
        output_file = os.path.join(self.output_dir, f"top_artists_{data_source}.html")
        fig.write_html(output_file)
        
        return {
            'chart_path': output_file,
            'top_artists': top_artists.to_dict(),
            'data_source': data_source,
            'title': title
        }
    
    def visualize_top_tracks(self, time_range='medium_term'):
        """Visualiza las canciones más escuchadas para un rango de tiempo específico."""
        # Determinar qué DataFrame usar
        if (self.top_tracks[time_range] is not None and 
            not self.top_tracks[time_range].empty and 
            'name' in self.top_tracks[time_range].columns):
            
            # Usar datos de top_tracks
            df = self.top_tracks[time_range]
            if 'rank' in df.columns:
                df = df.sort_values('rank')
            
            # Combinar nombre de canción y artista para la visualización
            if 'artist' in df.columns:
                df['track_artist'] = df['name'] + ' - ' + df['artist']
            else:
                df['track_artist'] = df['name']
                
            top_tracks = df['track_artist'].head(15).value_counts()
            
            range_name = {
                'short_term': 'último mes',
                'medium_term': 'últimos 6 meses',
                'long_term': 'de todos los tiempos'
            }
            
            title = f"Canciones más escuchadas ({range_name.get(time_range, time_range)})"
            data_source = f"top_tracks_{time_range}"
        else:
            # Usar el DataFrame más completo disponible
            df = self._get_best_dataframe()
            
            if df is None or df.empty or 'name' not in df.columns:
                return None
                
            # Combinar nombre de canción y artista para identificación única
            if 'artist' in df.columns:
                df['track_artist'] = df['name'] + ' - ' + df['artist']
            else:
                df['track_artist'] = df['name']
                
            top_tracks = df['track_artist'].value_counts().head(15)
            title = "Canciones más escuchadas"
            data_source = "combined_data"
        
        # Crear gráfico con Plotly
        fig = px.bar(
            x=top_tracks.values,
            y=top_tracks.index,
            orientation='h',
            labels={'x': 'Frecuencia', 'y': 'Canción'},
            title=title,
            color=top_tracks.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            width=800,
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar gráfico
        output_file = os.path.join(self.output_dir, f"top_tracks_{data_source}.html")
        fig.write_html(output_file)
        
        return {
            'chart_path': output_file,
            'top_tracks': top_tracks.to_dict(),
            'data_source': data_source,
            'title': title
        }
    
    def _get_best_dataframe(self):
        """Devuelve el DataFrame más completo disponible."""
        # Orden de preferencia: combined > current > recent
        if self.combined_df is not None and not self.combined_df.empty:
            return self.combined_df
        elif self.current_df is not None and not self.current_df.empty:
            return self.current_df
        elif self.recent_df is not None and not self.recent_df.empty:
            return self.recent_df
        else:
            return None
    
    def _get_best_dataframe_for_audio_features(self):
        """Devuelve el DataFrame con más atributos de audio."""
        # Buscar DataFrame con atributos de audio
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        dataframes = [
            ('combined', self.combined_df),
            ('current', self.current_df),
            ('recent', self.recent_df)
        ]
        
        best_df = None
        max_features = 0
        
        for name, df in dataframes:
            if df is None or df.empty:
                continue
                
            feature_count = sum(1 for feat in audio_features if feat in df.columns)
            
            if feature_count > max_features:
                max_features = feature_count
                best_df = df
        
        return best_df
    
    def _get_favorite_day(self):
        """Obtiene el día favorito para escuchar música."""
        df = self._get_best_dataframe()
        
        if df is None or df.empty or 'day_of_week' not in df.columns:
            if df is not None and 'datetime' in df.columns:
                df['day_of_week'] = df['datetime'].dt.day_name()
            else:
                return "No disponible"
        
        day_counts = df['day_of_week'].value_counts()
        favorite_day = day_counts.idxmax()
        
        # Traducir al español
        days_es = {
            'Monday': 'Lunes',
            'Tuesday': 'Martes',
            'Wednesday': 'Miércoles',
            'Thursday': 'Jueves',
            'Friday': 'Viernes',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
        
        return days_es.get(favorite_day, favorite_day)
    
    def _get_favorite_hour(self):
        """Obtiene la hora favorita para escuchar música."""
        df = self._get_best_dataframe()
        
        if df is None or df.empty:
            return "No disponible"
            
        if 'hour' not in df.columns and 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
        if 'hour' not in df.columns:
            return "No disponible"
            
        hour_counts = df['hour'].value_counts()
        favorite_hour = hour_counts.idxmax()
        
        # Formatear la hora en formato 12h
        if favorite_hour == 0:
            return "12 AM"
        elif favorite_hour < 12:
            return f"{favorite_hour} AM"
        elif favorite_hour == 12:
            return "12 PM"
        else:
            return f"{favorite_hour - 12} PM"
    
    def _get_top_artists_names(self, limit=3):
        """Obtiene los nombres de los artistas más escuchados."""
        df = self._get_best_dataframe()
        
        if df is None or df.empty or 'artist' not in df.columns:
            # Intentar usar top_artists si está disponible
            for time_range in ['short_term', 'medium_term', 'long_term']:
                if (self.top_artists[time_range] is not None and 
                    not self.top_artists[time_range].empty and 
                    'name' in self.top_artists[time_range].columns):
                    
                    top_artists = self.top_artists[time_range]['name'].head(limit).tolist()
                    return ", ".join(top_artists)
            
            return "No disponible"
        
        top_artists = df['artist'].value_counts().head(limit).index.tolist()
        return ", ".join(top_artists)
    
    def _get_mood_description(self):
        """Genera una descripción del estado de ánimo de la música."""
        df = self._get_best_dataframe_for_audio_features()
        
        if df is None or df.empty:
            return "No disponible"
        
        # Comprobar si tenemos atributos de audio
        mood_attrs = {
            'energy': 'energía',
            'valence': 'positividad',
            'danceability': 'ritmo bailable',
            'acousticness': 'acústica',
            'instrumentalness': 'instrumentalidad'
        }
        
        available_attrs = [attr for attr in mood_attrs if attr in df.columns]
        
        if not available_attrs:
            return "No disponible"
        
        # Calcular promedios
        mood_scores = {}
        for attr in available_attrs:
            mood_scores[attr] = df[attr].mean()
        
        # Determinar los atributos más destacados (más altos)
        sorted_attrs = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        top_attrs = sorted_attrs[:2]  # Los dos atributos más destacados
        
        mood_desc = []
        for attr, score in top_attrs:
            if score > 0.7:
                mood_desc.append(f"alta {mood_attrs[attr]}")
            elif score > 0.4:
                mood_desc.append(f"moderada {mood_attrs[attr]}")
            else:
                mood_desc.append(f"baja {mood_attrs[attr]}")
        
        return " y ".join(mood_desc)
    
    def analyze_genre_distribution(self):
        """
        Analiza la distribución de géneros musicales.
        Devuelve la ruta al archivo HTML generado.
        """
        # Buscar DataFrame con información de géneros
        genre_dfs = []
        
        # Intentar con top_artists que suele tener géneros
        for time_range in ['short_term', 'medium_term', 'long_term']:
            if (self.top_artists[time_range] is not None and 
                not self.top_artists[time_range].empty and 
                'genres' in self.top_artists[time_range].columns):
                genre_dfs.append(self.top_artists[time_range])
        
        # Intentar con otros DataFrames
        for df_name, df in [('combined', self.combined_df), ('current', self.current_df), ('recent', self.recent_df)]:
            if df is not None and not df.empty and 'genres' in df.columns:
                genre_dfs.append(df)
        
        if not genre_dfs:
            print("No hay datos de géneros disponibles")
            return None
        
        # Combinar todos los géneros
        all_genres = []
        
        for df in genre_dfs:
            for genres in df['genres'].dropna():
                if isinstance(genres, list):
                    all_genres.extend(genres)
                elif isinstance(genres, str):
                    # Intentar convertir string a lista (si está en formato JSON)
                    try:
                        genres_list = json.loads(genres.replace("'", '"'))
                        if isinstance(genres_list, list):
                            all_genres.extend(genres_list)
                    except:
                        # Si no es JSON, dividir por comas
                        all_genres.extend([g.strip() for g in genres.split(',')])
        
        # Contar frecuencia de géneros
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(15))
        
        # Crear gráfico
        fig = px.pie(
            names=list(top_genres.keys()),
            values=list(top_genres.values()),
            title='Distribución de géneros musicales',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=self.colors['secondary'],
            paper_bgcolor=self.colors['secondary'],
            font=dict(color=self.colors['accent'])
        )
        
        # Guardar gráfico
        output_file = os.path.join(self.output_dir, "genre_distribution.html")
        fig.write_html(output_file)
        
        print(f"Distribución de géneros guardada en: {output_file}")
        return output_file