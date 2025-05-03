import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import time
from typing import List, Dict, Any

class SpotifyHistoryRetriever:
    """
    Clase para recuperar el historial de escucha de Spotify y otras estadísticas.
    Utiliza la API oficial de Spotify para obtener:
    - Tus canciones más escuchadas (corto, medio y largo plazo)
    - Tus artistas más escuchados (corto, medio y largo plazo)
    - Recomendaciones basadas en tu historial
    - Tus canciones guardadas
    """

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, data_dir: str = "spotify_data"):
        """
        Inicializa el recolector de historial de Spotify.
        
        Args:
            client_id: ID de cliente de la API de Spotify
            client_secret: Secret de cliente de la API de Spotify
            redirect_uri: URI de redirección para autenticación OAuth
            data_dir: Directorio donde se almacenarán los datos
        """
        # Permisos necesarios para acceder al historial y datos personales
        scope = "user-read-recently-played user-top-read user-library-read user-follow-read playlist-read-private"
        
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
        
        # Verificar conexión
        try:
            self.user_info = self.sp.current_user()
            print(f"Conectado como: {self.user_info['display_name']} ({self.user_info['id']})")
        except Exception as e:
            print(f"Error al conectar con Spotify: {e}")
            self.user_info = None

    def get_recently_played(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene las canciones reproducidas recientemente.
        
        Args:
            limit: Número máximo de canciones a obtener (máximo 50 según API de Spotify)
            
        Returns:
            Lista de canciones reproducidas recientemente
        """
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            tracks = []
            
            for item in results['items']:
                track = item['track']
                played_at = item['played_at']  # Timestamp cuando se reprodujo
                
                track_info = {
                    'timestamp': played_at,
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
                tracks.append(track_info)
                
            # Guardar los datos
            filename = os.path.join(self.data_dir, "recently_played.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tracks, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado {len(tracks)} canciones reproducidas recientemente en {filename}")
            return tracks
            
        except Exception as e:
            print(f"Error al obtener canciones reproducidas recientemente: {e}")
            return []

    def get_top_tracks(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene tus canciones más escuchadas.
        
        Args:
            time_range: Período de tiempo para el análisis ('short_term': 4 semanas, 
                       'medium_term': 6 meses, 'long_term': varios años)
            limit: Número máximo de canciones a obtener (máximo 50)
            
        Returns:
            Lista de tus canciones más escuchadas
        """
        try:
            results = self.sp.current_user_top_tracks(time_range=time_range, limit=limit)
            tracks = []
            
            for i, track in enumerate(results['items'], 1):
                track_info = {
                    'rank': i,
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
                tracks.append(track_info)
                
            # Guardar los datos
            period_name = {
                'short_term': 'ultimo_mes',
                'medium_term': 'ultimos_6_meses',
                'long_term': 'todo_el_tiempo'
            }.get(time_range, time_range)
            
            filename = os.path.join(self.data_dir, f"top_tracks_{period_name}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tracks, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado tus {len(tracks)} canciones más escuchadas ({period_name}) en {filename}")
            return tracks
            
        except Exception as e:
            print(f"Error al obtener tus canciones más escuchadas: {e}")
            return []

    def get_top_artists(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene tus artistas más escuchados.
        
        Args:
            time_range: Período de tiempo para el análisis ('short_term': 4 semanas, 
                       'medium_term': 6 meses, 'long_term': varios años)
            limit: Número máximo de artistas a obtener (máximo 50)
            
        Returns:
            Lista de tus artistas más escuchados
        """
        try:
            results = self.sp.current_user_top_artists(time_range=time_range, limit=limit)
            artists = []
            
            for i, artist in enumerate(results['items'], 1):
                artist_info = {
                    'rank': i,
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres'],
                    'followers': artist['followers']['total'],
                    'popularity': artist['popularity'],
                    'url': artist['external_urls']['spotify'],
                    'image': artist['images'][0]['url'] if artist['images'] else None
                }
                artists.append(artist_info)
                
            # Guardar los datos
            period_name = {
                'short_term': 'ultimo_mes',
                'medium_term': 'ultimos_6_meses',
                'long_term': 'todo_el_tiempo'
            }.get(time_range, time_range)
            
            filename = os.path.join(self.data_dir, f"top_artists_{period_name}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(artists, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado tus {len(artists)} artistas más escuchados ({period_name}) en {filename}")
            return artists
            
        except Exception as e:
            print(f"Error al obtener tus artistas más escuchados: {e}")
            return []

    def get_saved_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene tus canciones guardadas en la biblioteca.
        
        Args:
            limit: Número máximo de canciones a obtener por solicitud
            
        Returns:
            Lista de tus canciones guardadas
        """
        try:
            tracks = []
            results = self.sp.current_user_saved_tracks(limit=limit)
            
            # Primera página de resultados
            for item in results['items']:
                track = item['track']
                added_at = item['added_at']  # Fecha cuando se añadió a favoritos
                
                track_info = {
                    'added_at': added_at,
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
                tracks.append(track_info)
            
            # Obtener el resto de páginas
            while results['next']:
                results = self.sp.next(results)
                for item in results['items']:
                    track = item['track']
                    added_at = item['added_at']
                    
                    track_info = {
                        'added_at': added_at,
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
                    tracks.append(track_info)
                
                # Evitar exceder el límite de solicitudes a la API
                time.sleep(0.1)
                
            # Guardar los datos
            filename = os.path.join(self.data_dir, "saved_tracks.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tracks, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado {len(tracks)} canciones de tu biblioteca en {filename}")
            return tracks
            
        except Exception as e:
            print(f"Error al obtener tus canciones guardadas: {e}")
            return []

    def get_playlists(self) -> List[Dict[str, Any]]:
        """
        Obtiene tus playlists y sus canciones.
        
        Returns:
            Lista de tus playlists con su información
        """
        try:
            # Obtener todas las playlists del usuario
            playlists_result = self.sp.current_user_playlists()
            user_playlists = []
            
            # Procesar cada playlist
            for playlist in playlists_result['items']:
                # Solo procesar playlists creadas o seguidas por el usuario
                playlist_info = {
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist['description'],
                    'owner': playlist['owner']['display_name'],
                    'owner_id': playlist['owner']['id'],
                    'public': playlist['public'],
                    'tracks_count': playlist['tracks']['total'],
                    'url': playlist['external_urls']['spotify'],
                    'image': playlist['images'][0]['url'] if playlist['images'] else None,
                    'tracks': []
                }
                
                # Obtener canciones de la playlist (máximo 100 por solicitud)
                tracks_result = self.sp.playlist_tracks(playlist['id'], limit=100)
                
                # Primera página de canciones
                for item in tracks_result['items']:
                    if item['track']:  # A veces hay elementos nulos
                        track = item['track']
                        track_info = {
                            'added_at': item['added_at'],
                            'added_by': item['added_by']['id'] if item['added_by'] else None,
                            'id': track['id'],
                            'name': track['name'],
                            'artist': ', '.join([artist['name'] for artist in track['artists']]),
                            'album': track['album']['name'] if 'album' in track else '',
                            'duration_ms': track['duration_ms']
                        }
                        playlist_info['tracks'].append(track_info)
                
                # Obtener el resto de páginas de canciones
                while tracks_result['next']:
                    tracks_result = self.sp.next(tracks_result)
                    for item in tracks_result['items']:
                        if item['track']:  # A veces hay elementos nulos
                            track = item['track']
                            track_info = {
                                'added_at': item['added_at'],
                                'added_by': item['added_by']['id'] if item['added_by'] else None,
                                'id': track['id'],
                                'name': track['name'],
                                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                                'album': track['album']['name'] if 'album' in track else '',
                                'duration_ms': track['duration_ms']
                            }
                            playlist_info['tracks'].append(track_info)
                    
                    # Evitar exceder el límite de solicitudes a la API
                    time.sleep(0.1)
                
                user_playlists.append(playlist_info)
                
            # Guardar los datos
            filename = os.path.join(self.data_dir, "playlists.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(user_playlists, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado {len(user_playlists)} playlists en {filename}")
            return user_playlists
            
        except Exception as e:
            print(f"Error al obtener tus playlists: {e}")
            return []

    def get_recommendations(self, seed_artists=None, seed_tracks=None, limit=20):
        """
        Obtiene recomendaciones basadas en tus artistas y canciones favoritas.
        
        Args:
            seed_artists: Lista de IDs de artistas para basar las recomendaciones (máx. 5)
            seed_tracks: Lista de IDs de canciones para basar las recomendaciones (máx. 5)
            limit: Número máximo de recomendaciones a obtener
            
        Returns:
            Lista de canciones recomendadas
        """
        try:
            # Si no se proporcionan semillas, usar los artistas y canciones más escuchados
            if not seed_artists and not seed_tracks:
                # Obtener top artistas y canciones si no se han proporcionado
                top_artists = self.get_top_artists(time_range='short_term', limit=5)
                top_tracks = self.get_top_tracks(time_range='short_term', limit=5)
                
                seed_artists = [artist['id'] for artist in top_artists[:2]]
                seed_tracks = [track['id'] for track in top_tracks[:3]]
            
            # Asegurarse de no exceder el límite de 5 semillas en total
            seed_artists = seed_artists[:5] if seed_artists else []
            seed_tracks = seed_tracks[:5-len(seed_artists)] if seed_tracks else []
            
            # Obtener recomendaciones
            recommendations = self.sp.recommendations(
                seed_artists=seed_artists,
                seed_tracks=seed_tracks,
                limit=limit
            )
            
            recommended_tracks = []
            for i, track in enumerate(recommendations['tracks'], 1):
                track_info = {
                    'rank': i,
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
                recommended_tracks.append(track_info)
            
            # Guardar los datos
            filename = os.path.join(self.data_dir, "recommendations.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(recommended_tracks, f, ensure_ascii=False, indent=2)
                
            print(f"Se han guardado {len(recommended_tracks)} canciones recomendadas en {filename}")
            return recommended_tracks
            
        except Exception as e:
            print(f"Error al obtener recomendaciones: {e}")
            return []

    def get_all_history(self):
        """
        Obtiene todos los datos históricos disponibles y los guarda.
        """
        print("Obteniendo historial completo de Spotify...")
        
        print("\n1. Obteniendo canciones reproducidas recientemente...")
        self.get_recently_played()
        
        print("\n2. Obteniendo tus canciones más escuchadas (último mes)...")
        self.get_top_tracks(time_range='short_term')
        
        print("\n3. Obteniendo tus canciones más escuchadas (últimos 6 meses)...")
        self.get_top_tracks(time_range='medium_term')
        
        print("\n4. Obteniendo tus canciones más escuchadas (todo el tiempo)...")
        self.get_top_tracks(time_range='long_term')
        
        print("\n5. Obteniendo tus artistas más escuchados (último mes)...")
        self.get_top_artists(time_range='short_term')
        
        print("\n6. Obteniendo tus artistas más escuchados (últimos 6 meses)...")
        self.get_top_artists(time_range='medium_term')
        
        print("\n7. Obteniendo tus artistas más escuchados (todo el tiempo)...")
        self.get_top_artists(time_range='long_term')
        
        print("\n8. Obteniendo tus canciones guardadas...")
        self.get_saved_tracks()
        
        print("\n9. Obteniendo tus playlists...")
        self.get_playlists()
        
        print("\n10. Obteniendo recomendaciones personalizadas...")
        self.get_recommendations()
        
        print("\n¡Todos los datos han sido descargados y guardados!")
        return True


if __name__ == "__main__":
    # Configuración de la API de Spotify
    # Debes obtener estos valores desde https://developer.spotify.com/dashboard
    CLIENT_ID = "tu_client_id"
    CLIENT_SECRET = "tu_client_secret"
    REDIRECT_URI = "http://localhost:8888/callback"
    
    # Inicializar el recolector de historial
    retriever = SpotifyHistoryRetriever(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI
    )
    
    # Obtener todo el historial disponible
    retriever.get_all_history()