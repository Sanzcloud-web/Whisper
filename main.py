#!/usr/bin/env python3
"""
Script principal pour la transcription en temps réel avec OpenAI Whisper
Usage: python main.py
"""

import json
import sys
import os
import pyaudio
import wave
import threading
import time
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import whisper
import ssl
import urllib.request

# Fix SSL pour macOS
ssl._create_default_https_context = ssl._create_unverified_context

class ContinuousAudioRecorder:
    """Enregistreur audio continu avec buffer circulaire"""
    
    def __init__(self, chunk_size=2048, sample_rate=44100, channels=1, overlap_duration=2):
        self.chunk_size = chunk_size          # Plus gros chunks pour moins de perte
        self.sample_rate = sample_rate        # Fréquence plus haute pour meilleure qualité
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.overlap_duration = overlap_duration
        
        # Buffer circulaire pour stockage continu
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
    def start_continuous_recording(self):
        """Démarre l'enregistrement continu en arrière-plan"""
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_continuously)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def _record_continuously(self):
        """Enregistre en continu dans un thread séparé"""
        try:
            # Tenter d'accéder au micro avec gestion de conflit
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None,  # Utilise le micro par défaut
                stream_callback=None
            )
            
            print("🔴 Enregistrement continu démarré...")
            print(f"📊 Config: {self.sample_rate}Hz, chunks={self.chunk_size}")
            
            while self.recording:
                try:
                    # Lecture non-bloquante avec gestion d'erreur
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if data:
                        self.audio_queue.put(data)
                except Exception as e:
                    print(f"⚠️ Erreur lecture audio: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Erreur ouverture stream: {e}")
            return
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            print("🔴 Enregistrement continu arrêté.")
    
    def get_audio_segment(self, duration=10):
        """Récupère un segment audio du buffer"""
        frames_needed = int(self.sample_rate / self.chunk_size * duration)
        frames = []
        empty_count = 0
        
        print(f"🎯 Récupération {frames_needed} frames pour {duration}s...")
        
        # Récupérer les frames du buffer avec diagnostic
        for i in range(frames_needed):
            try:
                frame = self.audio_queue.get(timeout=2.0)
                frames.append(frame)
                empty_count = 0  # Reset counter
            except queue.Empty:
                empty_count += 1
                if empty_count > 5:
                    print(f"⚠️ Buffer vide depuis {empty_count} tentatives, taille buffer: {self.audio_queue.qsize()}")
                time.sleep(0.1)
                continue
        
        if not frames:
            print("❌ Aucune frame récupérée!")
            return None
            
        print(f"✅ {len(frames)} frames récupérées sur {frames_needed} demandées")
            
        # Créer le dossier transcription s'il n'existe pas
        os.makedirs("transcription", exist_ok=True)
        
        # Sauvegarder le segment
        timestamp = datetime.now().strftime("%H%M%S")
        temp_file = f"transcription/segment_{timestamp}.wav"
        
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_file
    
    def stop_recording(self):
        """Arrête l'enregistrement continu"""
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
    
    def cleanup(self):
        """Nettoie les ressources"""
        self.stop_recording()
        self.audio.terminate()

class ParallelTranscriptionClient:
    """Client de transcription parallèle avec ThreadPoolExecutor"""
    
    def __init__(self, model="base", lang="fr", max_workers=2):
        self.model_name = model
        self.lang = lang
        self.session_start = datetime.now()
        self.whisper_model = None
        self.max_workers = max_workers
        
        # Créer le dossier transcription
        os.makedirs("transcription", exist_ok=True)
        
        # Nom de fichier unique avec timestamp
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.json_file = f"transcription/session_{session_id}.json"
        
        # ThreadPool pour transcriptions parallèles
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_transcriptions = {}  # Suivi des tâches
        
        # Lock pour accès concurrent au JSON
        self.json_lock = threading.Lock()
        
        # Initialiser le fichier JSON
        self.init_json_file()
        
        # Charger le modèle Whisper
        print(f"Chargement du modèle Whisper '{model}'...")
        self.whisper_model = whisper.load_model(model)
        print(f"Modèle chargé! ThreadPool: {max_workers} workers")
        
    def init_json_file(self):
        """Initialise le fichier JSON"""
        session_data = {
            "session": {
                "start_time": self.session_start.isoformat(),
                "session_id": self.session_start.strftime("%Y%m%d_%H%M%S"),
                "language": self.lang,
                "model": self.model_name,
                "status": "active"
            },
            "transcriptions": []
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"Fichier JSON initialisé: {self.json_file}")
    
    def _transcribe_single(self, audio_file):
        """Transcrit un seul fichier audio (worker thread)"""
        if self.whisper_model is None:
            return ""
        
        try:
            # Paramètres optimaux pour cours/amphithéâtre
            result = self.whisper_model.transcribe(
                audio_file, 
                language=self.lang,
                fp16=False,                # Force CPU mode
                logprob_threshold=-1.0,    # Valeur standard recommandée
                temperature=0.0,           # Déterministe pour cohérence
                compression_ratio_threshold=1.35,  # Optimisé pour voix classique
                condition_on_previous_text=True,  # Contexte pour cohérence
                word_timestamps=False,     # Plus rapide
                no_speech_threshold=0.2    # Équilibré pour voix prof
            )
            
            # Nettoyer le texte des artefacts
            text = result["text"].strip()
            
            # Filtrer les artefacts courants de Whisper
            artifacts = ['<|', '|>', '[BLANK_AUDIO]', '(SILENCE)', '...']
            for artifact in artifacts:
                text = text.replace(artifact, '')
            
            return text
                
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            return ""
    
    def transcribe_audio_async(self, audio_file, segment_id):
        """Lance une transcription en parallèle"""
        future = self.executor.submit(self._transcribe_single, audio_file)
        self.pending_transcriptions[segment_id] = {
            'future': future,
            'audio_file': audio_file,
            'timestamp': datetime.now()
        }
        return future
    
    def check_completed_transcriptions(self):
        """Vérifie et traite les transcriptions terminées"""
        completed = []
        
        for segment_id, task in self.pending_transcriptions.items():
            future = task['future']
            if future.done():
                try:
                    text = future.result()
                    timestamp = task['timestamp']
                    
                    # Sauvegarder le résultat
                    self.save_transcription(text, timestamp.isoformat())
                    
                    # Nettoyer le fichier audio
                    try:
                        os.remove(task['audio_file'])
                    except Exception as e:
                        print(f"⚠️ Erreur suppression {task['audio_file']}: {e}")
                    
                    completed.append(segment_id)
                    print(f"✅ Segment {segment_id} transcrit: '{text[:50]}...' 🗑️")
                    
                except Exception as e:
                    print(f"❌ Erreur récupération résultat {segment_id}: {e}")
                    completed.append(segment_id)
        
        # Nettoyer les tâches terminées
        for segment_id in completed:
            del self.pending_transcriptions[segment_id]
        
        return len(completed)
    
    def save_transcription(self, text, timestamp=None):
        """Sauvegarde thread-safe dans le JSON"""
        if text is None:
            text = ""
            
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        transcription_entry = {
            "timestamp": timestamp,
            "text": text.strip()
        }
        
        # Accès thread-safe au fichier JSON
        with self.json_lock:
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"session": {}, "transcriptions": []}
            
            data["transcriptions"].append(transcription_entry)
            
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def finalize_session(self):
        """Finalise la session et attend les transcriptions en cours"""
        print("🔄 Finalisation des transcriptions en cours...")
        
        # Attendre que toutes les transcriptions se terminent
        while self.pending_transcriptions:
            completed = self.check_completed_transcriptions()
            if completed > 0:
                print(f"⏳ {completed} transcriptions terminées, reste: {len(self.pending_transcriptions)}")
            time.sleep(0.5)
        
        # Fermer le ThreadPool
        self.executor.shutdown(wait=True)
        
        # Finaliser le JSON
        with self.json_lock:
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data["session"]["end_time"] = datetime.now().isoformat()
                data["session"]["status"] = "completed"
                data["session"]["total_transcriptions"] = len(data["transcriptions"])
                
                with open(self.json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"\n✅ Session finalisée. {len(data['transcriptions'])} transcriptions sauvegardées.")
                
            except Exception as e:
                print(f"❌ Erreur finalisation: {e}")

def cleanup_temp_files():
    """Nettoie les fichiers temporaires au démarrage"""
    if os.path.exists("transcription"):
        temp_files = [f for f in os.listdir("transcription") if f.startswith("segment_") and f.endswith(".wav")]
        if temp_files:
            print(f"🗑️ Nettoyage de {len(temp_files)} fichier(s) temporaire(s)...")
            for file in temp_files:
                try:
                    os.remove(f"transcription/{file}")
                except Exception as e:
                    print(f"Erreur suppression {file}: {e}")

def main():
    """
    Lance la transcription en temps réel avec OpenAI Whisper
    """
    print("Initialisation de la transcription Whisper avec sauvegarde JSON...")
    
    # Nettoyer les fichiers temporaires
    cleanup_temp_files()
    
    try:
        # Créer les clients avec transcription PARALLÈLE
        transcription_client = ParallelTranscriptionClient(model="large", lang="fr", max_workers=2)
        audio_recorder = ContinuousAudioRecorder()
        
        print("Configuration terminée. Démarrage de la transcription PARALLÈLE...")
        print("Enregistrement CONTINU + Transcription PARALLÈLE = VITESSE MAX!")
        print("Ctrl+C pour arrêter.")
        print(f"📁 Dossier: transcription/")
        print(f"📄 Fichier JSON: {transcription_client.json_file}")
        print("🚀 2 threads de transcription en parallèle")
        print("Segments analysés toutes les 3 secondes...")
        print("-" * 50)
        
        # Démarrer l'enregistrement continu
        audio_recorder.start_continuous_recording()
        time.sleep(2)  # Laisser le buffer se remplir
        
        segment_counter = 0
        
        # Boucle de transcription ULTRA-RAPIDE
        while True:
            try:
                # Vérifier les transcriptions terminées
                completed = transcription_client.check_completed_transcriptions()
                
                # Récupérer un nouveau segment
                print(f"📊 Segment #{segment_counter}...", end="", flush=True)
                temp_audio = audio_recorder.get_audio_segment(duration=10)
                
                if temp_audio:
                    # Lancer transcription en PARALLÈLE (non-bloquant!)
                    print(" 🚀 Lancé en parallèle...", end="", flush=True)
                    transcription_client.transcribe_audio_async(temp_audio, segment_counter)
                    print(" ✅")
                    segment_counter += 1
                else:
                    print(" ⚠️ (pas de données)")
                
                # Délai RÉDUIT pour vitesse maximale
                time.sleep(3)  # Analyse toutes les 3 secondes
                
            except Exception as e:
                print(f"\nErreur dans la boucle: {e}")
                time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nTranscription arrêtée par l'utilisateur.")
        if 'transcription_client' in locals():
            transcription_client.finalize_session()
        if 'audio_recorder' in locals():
            audio_recorder.cleanup()
        sys.exit(0)
        
    except Exception as e:
        print(f"Erreur: {e}")
        print("\nAssurez-vous d'avoir installé:")
        print("pip install openai-whisper pyaudio")
        if 'transcription_client' in locals():
            transcription_client.finalize_session()
        if 'audio_recorder' in locals():
            audio_recorder.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
