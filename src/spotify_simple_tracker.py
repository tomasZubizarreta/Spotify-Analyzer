import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import json
import os
from datetime import datetime

class SimpleSpotifyTracker:
    def __init__(self, client_id, client_secret, redirect_uri, scope, data_dir="./spotify_data"):
        """
        Inicializa una versión simplificada del tracker de Spotify.
        
        Args:
            client_id: ID de cliente de la API de Spotify
            client_secret: Secret de cliente de la API de Spotify
            redirect_uri: URI de redirección para autenticación OAuth
            scope: Permisos solicitados a Spotify
            data_dir: Directorio donde se almacenarán los datos
        """
        # Configuración de autenticación
        self.auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            open_browser=True
        )
        
        # Crear cliente Spotify
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        
        # Configuración de almacenamiento
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.track_history_file = os.path.join(self.data_dir, "track_history.json")
        self.current_track_id = None
        self.track_history = self.load_track_history()
        
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
        """Obtiene la información básica de la canción que se está reproduciendo actualmente."""
        try:
            # Obtener la reproducción actual
            current_playback = self.sp.current_playback()
            
            # Verificar si hay algo reproduciéndose
            if not current_playback or not current_playback.get('item'):
                print("No se está reproduciendo nada en este momento.")
                return None
                
            track = current_playback['item']
            
            # Verificar si la canción ya fue registrada (evitar duplicados)
            if track['id'] == self.current_track_id:
                return None
                
            self.current_track_id = track['id']
            
            # Formatear la información básica de la canción
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
                'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None
            }
            
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
    
    def start_tracking(self, interval=30):
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
    
    def print_top_items(self, n=5):
        """Muestra los artistas y canciones más escuchados."""
        if not self.track_history:
            print("No hay suficientes datos para mostrar estadísticas.")
            return
            
        artists = {}
        songs = {}
        
        for track in self.track_history:
            # Contar artistas
            artist = track['artist']
            artists[artist] = artists.get(artist, 0) + 1
            
            # Contar canciones
            song = f"{track['name']} - {track['artist']}"
            songs[song] = songs.get(song, 0) + 1
        
        # Mostrar top artistas
        print(f"\nTOP {n} ARTISTAS:")
        top_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, (artist, count) in enumerate(top_artists, 1):
            print(f"{i}. {artist}: {count} reproducciones")
        
        # Mostrar top canciones
        print(f"\nTOP {n} CANCIONES:")
        top_songs = sorted(songs.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, (song, count) in enumerate(top_songs, 1):
            print(f"{i}. {song}: {count} reproducciones")


if __name__ == "__main__":
    # Configuración de la API de Spotify
    # Debes obtener estos valores desde https://developer.spotify.com/dashboard
    CLIENT_ID = "eac4fe1c81214c4a9529be3d2bc26088"
    CLIENT_SECRET = "7f17c3702e314d059ad16b70dce94c59"
    REDIRECT_URI = "http://localhost:8888/callback"
    SCOPE = "user-read-currently-playing user-read-playback-state"
    
    # Inicializar el tracker
    tracker = SimpleSpotifyTracker(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )
    
    # Ejemplo: mostrar estadísticas existentes
    tracker.print_top_items()
    
    # Iniciar el seguimiento continuo
    tracker.start_tracking(interval=30)  # Verificar cada 30 segundos