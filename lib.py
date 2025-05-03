import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter

class SpotifyAnalyzer:
    def __init__(self, data_file="spotify_data/track_history.json"):
        """
        Inicializa el analizador de datos de Spotify.
        
        Args:
            data_file: Ruta al archivo JSON con los datos de Spotify
        """
        self.data_file = data_file
        self.tracks_df = None
        self.load_data()
        
    def load_data(self):
        """Carga los datos desde el archivo JSON a un DataFrame."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                tracks = json.load(f)
                
            # Convertir a DataFrame
            self.tracks_df = pd.DataFrame(tracks)
            
            # Convertir timestamp a datetime
            self.tracks_df['datetime'] = pd.to_datetime(self.tracks_df['timestamp'])
            
            # Añadir columnas de fecha y hora
            self.tracks_df['date'] = self.tracks_df['datetime'].dt.date
            self.tracks_df['hour'] = self.tracks_df['datetime'].dt.hour
            self.tracks_df['day_of_week'] = self.tracks_df['datetime'].dt.day_name()
            
            print(f"Datos cargados: {len(self.tracks_df)} canciones")
            
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
    
    def get_listening_patterns(self):
        """Analiza patrones de escucha por día y hora."""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return "No hay datos suficientes para analizar."
            
        # Patrones por día de la semana
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.tracks_df['day_of_week'].value_counts().reindex(days_order)
        
        # Patrones por hora del día
        hour_counts = self.tracks_df['hour'].value_counts().sort_index()
        
        plt.figure(figsize=(15, 10))
        
        # Gráfico por día
        plt.subplot(2, 1, 1)
        sns.barplot(x=day_counts.index, y=day_counts.values)
        plt.title('Canciones por día de la semana')
        plt.xlabel('Día')
        plt.ylabel('Número de canciones')
        
        # Gráfico por hora
        plt.subplot(2, 1, 2)
        sns.barplot(x=hour_counts.index, y=hour_counts.values)
        plt.title('Canciones por hora del día')
        plt.xlabel('Hora')
        plt.ylabel('Número de canciones')
        plt.xticks(range(0, 24))
        
        plt.tight_layout()
        
        # Guardar gráfico
        output_file = "spotify_data/listening_patterns.png"
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def analyze_mood_patterns(self):
        """Analiza patrones de estado de ánimo basados en características de audio."""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return "No hay datos suficientes para analizar."
            
        # Seleccionar características relacionadas con el estado de ánimo
        mood_features = ['valence', 'energy', 'danceability', 'acousticness']
        
        # Asegurarse de que todas las columnas existen
        if not all(feature in self.tracks_df.columns for feature in mood_features):
            return "Faltan características de audio necesarias para este análisis."
            
        # Crear un DataFrame solo con estas características
        mood_df = self.tracks_df[mood_features + ['name', 'artist', 'datetime']].dropna()
        
        if len(mood_df) < 10:  # Necesitamos un mínimo de datos para el análisis
            return "No hay suficientes datos con características de audio para analizar."
            
        # Agrupar por fecha para ver la evolución del estado de ánimo
        daily_mood = mood_df.groupby(mood_df['datetime'].dt.date)[mood_features].mean()
        
        plt.figure(figsize=(15, 8))
        daily_mood.plot(figsize=(15, 8))
        plt.title('Evolución diaria del estado de ánimo musical')
        plt.xlabel('Fecha')
        plt.ylabel('Valor (0-1)')
        plt.legend(title='Característica')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar gráfico
        output_file = "spotify_data/mood_evolution.png"
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def cluster_music_taste(self, n_clusters=4):
        """
        Agrupa canciones en clusters basados en sus características de audio.
        
        Args:
            n_clusters: Número de clusters a crear
        """
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return "No hay datos suficientes para analizar."
            
        # Características para clustering
        cluster_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo'
        ]
        
        # Asegurarse de que todas las columnas existen
        available_features = [f for f in cluster_features if f in self.tracks_df.columns]
        
        if len(available_features) < 3:  # Necesitamos al menos algunas características
            return "No hay suficientes características de audio para realizar clustering."
            
        # Crear un DataFrame para clustering
        cluster_df = self.tracks_df[available_features + ['name', 'artist']].dropna()
        
        if len(cluster_df) < n_clusters * 2:  # Necesitamos datos suficientes
            return "No hay suficientes canciones con datos completos para clustering."
            
        # Normalizar datos para clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_df[available_features])
        
        # Realizar clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Analizar clusters
        cluster_stats = cluster_df.groupby('cluster')[available_features].mean()
        
        # Encontrar canciones representativas de cada cluster
        representative_songs = []
        for cluster_id in range(n_clusters):
            cluster_songs = cluster_df[cluster_df['cluster'] == cluster_id]
            if len(cluster_songs) > 0:
                # Encontrar el punto central del cluster
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Calcular distancia de cada canción al centro
                distances = []
                for idx, song in cluster_songs.iterrows():
                    song_features = song[available_features].values
                    distance = np.linalg.norm(song_features - cluster_center)
                    distances.append((idx, distance))
                
                # Ordenar por distancia y obtener las 3 más cercanas
                closest_songs = sorted(distances, key=lambda x: x[1])[:3]
                
                for song_idx, _ in closest_songs:
                    song = cluster_df.loc[song_idx]
                    representative_songs.append({
                        'cluster': cluster_id,
                        'name': song['name'],
                        'artist': song['artist']
                    })
        
        # Visualizar clusters (PCA para reducir dimensionalidad)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        plt.figure(figsize=(12, 8))
        for cluster_id in range(n_clusters):
            cluster_points = pca_result[cluster_df['cluster'] == cluster_id]
            if len(cluster_points) > 0:
                plt.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1], 
                    label=f'Cluster {cluster_id}'
                )
        
        plt.title('Clusters de gustos musicales')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Guardar gráfico
        output_file = "spotify_data/music_clusters.png"
        plt.savefig(output_file)
        plt.close()
        
        return {
            'chart_file': output_file,
            'cluster_stats': cluster_stats.to_dict(),
            'representative_songs': representative_songs
        }
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en tus artistas y canciones más escuchadas."""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return "No hay datos suficientes para generar recomendaciones."
            
        try:
            # Obtener los artistas y canciones más frecuentes
            top_artists = Counter(self.tracks_df['artist']).most_common(5)
            top_tracks = Counter(zip(self.tracks_df['name'], self.tracks_df['artist'])).most_common(5)
            
            # Obtener IDs únicos de canciones para evitar repeticiones
            unique_track_ids = list(set(self.tracks_df['id'].dropna()))
            
            # Esta función requiere una conexión a la API de Spotify
            # Aquí solo se muestra cómo se estructuraría
            print("Para implementar recomendaciones reales, necesitarías:")
            print("1. Conectar con la API de Spotify")
            print("2. Usar los endpoints de recomendación basados en tus top tracks/artists")
            
            recommendations = {
                'based_on_top_artists': top_artists,
                'based_on_top_tracks': top_tracks,
                'note': "Para obtener recomendaciones reales, necesitas implementar la conexión a la API de Spotify"
            }
            
            return recommendations
            
        except Exception as e:
            return f"Error al generar recomendaciones: {e}"
    
    def create_dashboard(self):
        """Crea un dashboard completo con todos los análisis."""
        os.makedirs("spotify_data/dashboard", exist_ok=True)
        
        # Recopilar estadísticas básicas
        total_tracks = len(self.tracks_df) if self.tracks_df is not None else 0
        unique_artists = len(self.tracks_df['artist'].unique()) if self.tracks_df is not None else 0
        listening_time = self.tracks_df['duration_ms'].sum() / (1000 * 60 * 60) if self.tracks_df is not None else 0
        
        stats = {
            'total_tracks': total_tracks,
            'unique_artists': unique_artists,
            'total_listening_hours': round(listening_time, 2),
            'first_record': str(self.tracks_df['datetime'].min()) if total_tracks > 0 else "N/A",
            'last_record': str(self.tracks_df['datetime'].max()) if total_tracks > 0 else "N/A"
        }
        
        # Generar todos los análisis
        patterns_chart = self.get_listening_patterns()
        mood_chart = self.analyze_mood_patterns()
        clusters = self.cluster_music_taste()
        
        # Crear un informe HTML simple
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mi Dashboard de Spotify</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats {{ display: flex; justify-content: space-between; margin-bottom: 30px; }}
                .stat-box {{ background-color: #1DB954; color: white; padding: 20px; border-radius: 10px; flex: 1; margin: 0 10px; text-align: center; }}
                .chart-section {{ margin-bottom: 40px; }}
                h1, h2 {{ color: #1DB954; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Mi Dashboard de Spotify</h1>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>{stats['total_tracks']}</h3>
                        <p>Canciones escuchadas</p>
                    </div>
                    <div class="stat-box">
                        <h3>{stats['unique_artists']}</h3>
                        <p>Artistas únicos</p>
                    </div>
                    <div class="stat-box">
                        <h3>{stats['total_listening_hours']}</h3>
                        <p>Horas de escucha</p>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h2>Patrones de escucha</h2>
                    <img src="../listening_patterns.png" width="100%" />
                </div>
                
                <div class="chart-section">
                    <h2>Evolución del estado de ánimo musical</h2>
                    <img src="../mood_evolution.png" width="100%" />
                </div>
                
                <div class="chart-section">
                    <h2>Clusters de gustos musicales</h2>
                    <img src="../music_clusters.png" width="100%" />
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar el dashboard
        dashboard_file = "spotify_data/dashboard/index.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return dashboard_file
        

if __name__ == "__main__":
    analyzer = SpotifyAnalyzer("spotify_data/track_history.json")
    dashboard_file = analyzer.create_dashboard()
    print(f"Dashboard creado en: {dashboard_file}")