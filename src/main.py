import os
import json
import configparser
from datetime import datetime
from spotify_simple_tracker import SimpleSpotifyTracker
from spotify_history_retriever import SpotifyHistoryRetriever
from lib import SpotifyAnalyzer

class SpotifyManager:
    """
    Clase principal que integra todas las funcionalidades de Spotify:
    - Seguimiento en tiempo real de canciones (SimpleSpotifyTracker)
    - Descarga del historial completo (SpotifyHistoryRetriever)
    - Análisis de datos y visualizaciones (lib)
    """
    
    def __init__(self):
        # Configuración de la API de Spotify
        config = configparser.ConfigParser()
        config.read("./config.ini")

        self.client_id = config.get("API_Keys", "client_id")  # Reemplaza con tu client_id
        self.client_secret = config.get("API_Keys", "client_secret")  # Reemplaza con tu client_secret
        self.redirect_uri = "http://localhost:8888/callback"
        self.data_dir = "spotify_data"
        
        # Asegurar que existe el directorio de datos
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Inicializar componentes
        self.tracker = None
        self.history_retriever = None
        self.analyzer = None
        
    def setup_tracker(self):
        """Configura y devuelve el rastreador de canciones en tiempo real"""
        scope = "user-read-currently-playing user-read-playback-state"
        self.tracker = SimpleSpotifyTracker(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=scope,
            data_dir=self.data_dir
        )
        return self.tracker
        
    def setup_history_retriever(self):
        """Configura y devuelve el recolector de historial"""
        self.history_retriever = SpotifyHistoryRetriever(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            data_dir=self.data_dir
        )
        return self.history_retriever
        
    def setup_analyzer(self, data_file=None):
        """Configura y devuelve el analizador de datos"""
        if data_file is None:
            data_file = os.path.join(self.data_dir, "track_history.json")
        self.analyzer = SpotifyAnalyzer(data_file=data_file, data_dir="spotify_data")
        return self.analyzer
    
    def start_realtime_tracking(self, interval=30):
        """Inicia el seguimiento en tiempo real de canciones"""
        if not self.tracker:
            self.setup_tracker()
        print(f"Iniciando seguimiento en tiempo real cada {interval} segundos...")
        self.tracker.start_tracking(interval=interval)
    
    def download_complete_history(self):
        """Descarga todo el historial disponible desde Spotify"""
        if not self.history_retriever:
            self.setup_history_retriever()
        print("Descargando historial completo de Spotify...")
        self.history_retriever.get_all_history()
    
    def analyze_data(self):
        """Analiza los datos recopilados y crea visualizaciones"""
        if not self.analyzer:
            self.setup_analyzer()
        
        print("Analizando datos...")
        
        print("Generando dashboard...")
        dashboard_file = self.analyzer.create_dashboard()
        
        print(f"\nAnálisis completo. Dashboard disponible en: {dashboard_file}")
        return dashboard_file
    
    def merge_history_files(self):
        """
        Combina varios archivos de historial en uno solo para un análisis más completo
        """
        all_tracks = []
        file_paths = [
            os.path.join(self.data_dir, "track_history.json"),
            os.path.join(self.data_dir, "recently_played.json"),
            os.path.join(self.data_dir, "top_tracks_ultimo_mes.json"),
            os.path.join(self.data_dir, "top_tracks_ultimos_6_meses.json"),
            os.path.join(self.data_dir, "top_tracks_todo_el_tiempo.json")
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tracks = json.load(f)
                        print(f"Cargados {len(tracks)} tracks desde {file_path}")
                        
                        # Asegurarse de que cada track tenga un timestamp si no lo tiene
                        for track in tracks:
                            if 'timestamp' not in track:
                                if 'added_at' in track:
                                    track['timestamp'] = track['added_at']
                                else:
                                    track['timestamp'] = datetime.now().isoformat()
                        
                        all_tracks.extend(tracks)
                except Exception as e:
                    print(f"Error al cargar {file_path}: {e}")
        
        # Eliminar duplicados basados en el ID de la canción
        unique_tracks = {}
        for track in all_tracks:
            if 'id' in track and track['id'] not in unique_tracks:
                unique_tracks[track['id']] = track
        
        combined_tracks = list(unique_tracks.values())
        
        # Guardar el archivo combinado
        combined_file = os.path.join(self.data_dir, "all_tracks_combined.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_tracks, f, ensure_ascii=False, indent=2)
            
        print(f"Se han combinado {len(combined_tracks)} tracks únicos en {combined_file}")
        return combined_file
    
    def run_complete_analysis(self):
        """
        Ejecuta todo el proceso:
        1. Descarga el historial completo
        2. Combina todos los archivos de historial
        3. Analiza los datos combinados
        """
        # Paso 1: Descargar historial
        self.download_complete_history()
        
        # Paso 2: Combinar archivos
        combined_file = self.merge_history_files()
        
        # Paso 3: Analizar datos combinados
        self.analyzer = SpotifyAnalyzer(data_file=combined_file, data_dir="spotify_data")
        dashboard_file = self.analyze_data()
        
        return dashboard_file

def print_menu():
    """Muestra el menú de opciones"""
    print("\n==== SPOTIFY MANAGER ====")
    print("1. Iniciar seguimiento en tiempo real")
    print("2. Descargar historial completo de Spotify")
    print("3. Analizar datos existentes")
    print("4. Análisis completo (descarga + análisis)")

    print("6. Salir")
    print("========================")
    return input("Selecciona una opción (1-5): ")

if __name__ == "__main__":
    # Crear el gestor de Spotify
    manager = SpotifyManager()
    
    while True:
        option = print_menu()
        
        if option == '1':
            interval = int(input("Intervalo de seguimiento en segundos (por defecto 30): ") or "30")
            manager.start_realtime_tracking(interval)
        elif option == '2':
            manager.download_complete_history()
        elif option == '3':
            dashboard_file = manager.analyze_data()
            print(f"Dashboard creado en: {dashboard_file}")
        elif option == '4':
            dashboard_file = manager.run_complete_analysis()
            print(f"Análisis completo terminado. Dashboard creado en: {dashboard_file}")
        elif option == '5':
            analyzer = SpotifyAnalyzer(data_file="ruta/a/tus_datos.json", data_dir="spotify_data")
            # Patrones de escucha
            analyzer.get_listening_patterns()

            # Análisis de estado de ánimo
            analyzer.analyze_mood_patterns()

            # Agrupación de canciones por estilo
            analyzer.cluster_music_taste()

            dashboard_file = analyzer.create_dashboard() ## TODO
            print(f"Dashboard creado en: {dashboard_file}")

        elif option == '6':
            print("¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")