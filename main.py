import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import json
import os
from datetime import datetime

class SpotifyTracker:
    def __init__(self, client_id, client_secret, redirect_uri, scope, data_dir="spotify_data"):
        """
        Inicializa el tracker de Spotify.
        
        Args:
            client_id: ID de cliente de la API de Spotify
            client_secret: Secret de cliente de la API de Spotify
            redirect_uri: URI de redirección para autenticación OAuth
            scope: Permisos solicitados a Spotify
            data_dir: Directorio donde se almacenarán los datos
        """
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        ))
        
        self.data_dir = data_dir
        self.ensure_data_directory()
        self.track_history_file = os.path.join(self.data_dir, "track_history.json")
        self.current_track_id = None
        self.track_history = self.load_track_history()
        
    def ensure_data_directory(self):
        """Asegura que el directorio de datos exista."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def load_track_history(self):
        """Carga el historial de canciones desde el archivo local."""
        if os.path.exists(self.track_history_file):
            try:
                with open(self.track_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error al leer el archivo de historial, creando uno nuevo.")
                return []
        return []
    
    def save_track_history(self):
        """Guarda el historial de canciones en el archivo local."""
        with open(self.track_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.track_history, f, ensure_ascii=False, indent=2)
            
    def get_current_track(self):
        """Obtiene la información de la canción que se está reproduciendo actualmente."""
        try:
            current_playback = self.sp.current_playback()
            
            if not current_playback or not current_playback.get('item'):
                return None
                
            track = current_playback['item']
            
            # Comprobar si la canción es diferente a la última registrada
            if track['id'] == self.current_track_id:
                return None
                
            self.current_track_id = track['id']
            
            # Obtener detalles de audio de la canción
            audio_features = self.sp.audio_features(track['id'])[0] if track['id'] else None
            
            # Formatear la información de la canción
            track_info = {
                'timestamp': datetime.now().isoformat(),
                'id': track['id'],
                'name': track['name'],
                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'url': track['external_urls']['spotify'],
                'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'preview_url': track['preview_url'],
                'playing_device': current_playback.get('device', {}).get('name', 'Unknown')
            }
            
            # Añadir características de audio si están disponibles
            if audio_features:
                audio_data = {
                    'acousticness': audio_features.get('acousticness'),
                    'danceability': audio_features.get('danceability'),
                    'energy': audio_features.get('energy'),
                    'instrumentalness': audio_features.get('instrumentalness'),
                    'key': audio_features.get('key'),
                    'liveness': audio_features.get('liveness'),
                    'loudness': audio_features.get('loudness'),
                    'mode': audio_features.get('mode'),
                    'speechiness': audio_features.get('speechiness'),
                    'tempo': audio_features.get('tempo'),
                    'time_signature': audio_features.get('time_signature'),
                    'valence': audio_features.get('valence')
                }
                track_info.update(audio_data)
                
            return track_info
            
        except Exception as e:
            print(f"Error al obtener la canción actual: {e}")
            return None
            
    def track_current_song(self):
        """Registra la canción actual y la guarda en el historial."""
        track_info = self.get_current_track()
        
        if track_info:
            print(f"Escuchando ahora: {track_info['name']} - {track_info['artist']}")
            self.track_history.append(track_info)
            self.save_track_history()
            return track_info
        return None
    
    def start_tracking(self, interval=60):
        """
        Inicia el seguimiento continuo de las canciones reproducidas.
        
        Args:
            interval: Tiempo en segundos entre cada verificación
        """
        print(f"Iniciando seguimiento de Spotify cada {interval} segundos. Presiona Ctrl+C para detener.")
        try:
            while True:
                self.track_current_song()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nSeguimiento detenido.")
            
    def get_statistics(self):
        """Genera estadísticas básicas sobre el historial de escucha."""
        if not self.track_history:
            return "No hay datos suficientes para generar estadísticas."
            
        artists = {}
        albums = {}
        songs = {}
        
        for track in self.track_history:
            # Contar artistas
            artist = track['artist']
            artists[artist] = artists.get(artist, 0) + 1
            
            # Contar álbumes
            album = track['album']
            albums[album] = albums.get(album, 0) + 1
            
            # Contar canciones
            song = f"{track['name']} - {track['artist']}"
            songs[song] = songs.get(song, 0) + 1
        
        # Ordenar por frecuencia
        top_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:5]
        top_albums = sorted(albums.items(), key=lambda x: x[1], reverse=True)[:5]
        top_songs = sorted(songs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        stats = {
            "total_tracks": len(self.track_history),
            "unique_artists": len(artists),
            "unique_albums": len(albums),
            "unique_songs": len(songs),
            "top_artists": top_artists,
            "top_albums": top_albums,
            "top_songs": top_songs
        }
        
        return stats
        
    def export_data(self, format="json"):
        """
        Exporta los datos a diferentes formatos.
        
        Args:
            format: Formato de exportación ("json" por defecto)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = os.path.join(self.data_dir, f"spotify_export_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.track_history, f, ensure_ascii=False, indent=2)
            return filename
        else:
            raise ValueError(f"Formato de exportación '{format}' no soportado")


if __name__ == "__main__":
    # Configuración de la API de Spotify
    # Debes obtener estos valores desde https://developer.spotify.com/dashboard
    CLIENT_ID = "client-id"
    CLIENT_SECRET = "client-secret"
    REDIRECT_URI = "http://localhost:8888/callback"
    SCOPE = "user-read-currently-playing user-read-playback-state"
    
    tracker = SpotifyTracker(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )
    
    # Iniciar el seguimiento continuo
    tracker.start_tracking(interval=30)  # Verificar cada 30 segundos