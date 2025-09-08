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
import whisper
import ssl
import urllib.request

# Fix SSL pour macOS
ssl._create_default_https_context = ssl._create_unverified_context

class ContinuousAudioRecorder:
    """Enregistreur audio continu avec buffer circulaire"""
    
    def __init__(self, chunk_size=1024, sample_rate=22050, channels=1, overlap_duration=2):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.overlap_duration = overlap_duration  # Chevauchement en secondes
        
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
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🔴 Enregistrement continu démarré...")
        
        while self.recording:
            try:
                data = stream.read(self.chunk_size)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Erreur enregistrement: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        print("🔴 Enregistrement continu arrêté.")
    
    def get_audio_segment(self, duration=10):
        """Récupère un segment audio du buffer"""
        frames_needed = int(self.sample_rate / self.chunk_size * duration)
        frames = []
        
        # Récupérer les frames du buffer
        for _ in range(frames_needed):
            try:
                frame = self.audio_queue.get(timeout=1.0)
                frames.append(frame)
            except queue.Empty:
                print("⚠️  Buffer audio vide, attente...")
                time.sleep(0.1)
                continue
        
        if not frames:
            return None
            
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

class JSONTranscriptionClient:
    """Client pour la transcription avec sauvegarde JSON"""
    
    def __init__(self, model="base", lang="fr"):
        self.model_name = model
        self.lang = lang
        self.json_file = "transcription.json"
        self.session_start = datetime.now().isoformat()
        self.whisper_model = None
        
        # Initialiser le fichier JSON
        self.init_json_file()
        
        # Charger le modèle Whisper
        print(f"Chargement du modèle Whisper '{model}'...")
        self.whisper_model = whisper.load_model(model)
        print("Modèle chargé!")
        
    def init_json_file(self):
        """Initialise le fichier JSON"""
        session_data = {
            "session": {
                "start_time": self.session_start,
                "language": self.lang,
                "model": self.model_name,
                "status": "active"
            },
            "transcriptions": []
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"Fichier JSON initialisé: {self.json_file}")
    
    def transcribe_audio(self, audio_file):
        """Transcrit un fichier audio"""
        if self.whisper_model is None:
            return ""
        
        try:
            # Options optimisées pour amphithéâtre
            result = self.whisper_model.transcribe(
                audio_file, 
                language=self.lang,
                fp16=False,                # Force CPU mode
                no_speech_threshold=0.4,   # Plus sensible pour voix lointaine
                logprob_threshold=-1.0,    # Filtre les prédictions peu fiables
                temperature=0.0,           # Pas de randomness
                compression_ratio_threshold=2.4,  # Améliore la robustesse
                condition_on_previous_text=True,  # Contexte pour cohérence
                word_timestamps=False       # Pas de timestamps de mots (plus rapide)
            )
            
            # Nettoyer le texte des artefacts
            text = result["text"].strip()
            
            # Filtrer les artefacts courants de Whisper
            artifacts = ['<|', '|>', '[BLANK_AUDIO]', '(SILENCE)', '...']
            for artifact in artifacts:
                text = text.replace(artifact, '')
            
            # Retourner seulement si le texte est significatif
            if len(text) > 2 and not text.isspace():
                return text
            else:
                return ""
                
        except Exception as e:
            print(f"Erreur transcription: {e}")
            return ""
    
    def save_transcription(self, text, timestamp=None):
        """Sauvegarde dans le JSON"""
        if not text or not text.strip():
            return
            
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        transcription_entry = {
            "timestamp": timestamp,
            "text": text.strip()
        }
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"session": {}, "transcriptions": []}
        
        data["transcriptions"].append(transcription_entry)
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[{timestamp[:19]}] {text}")
    
    def finalize_session(self):
        """Finalise la session"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["session"]["end_time"] = datetime.now().isoformat()
            data["session"]["status"] = "completed"
            data["session"]["total_transcriptions"] = len(data["transcriptions"])
            
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"\nSession finalisée. {len(data['transcriptions'])} transcriptions sauvegardées.")
            
        except Exception as e:
            print(f"Erreur finalisation: {e}")

def main():
    """
    Lance la transcription en temps réel avec OpenAI Whisper
    """
    print("Initialisation de la transcription Whisper avec sauvegarde JSON...")
    
    try:
        # Créer les clients avec modèle large pour amphithéâtre
        transcription_client = JSONTranscriptionClient(model="large", lang="fr")
        audio_recorder = ContinuousAudioRecorder()
        
        print("Configuration terminée. Démarrage de la transcription...")
        print("Enregistrement CONTINU - aucun audio perdu!")
        print("Ctrl+C pour arrêter.")
        print(f"Transcriptions sauvegardées dans: {transcription_client.json_file}")
        print("Segments analysés toutes les 8 secondes (optimisé amphithéâtre)...")
        print("-" * 50)
        
        # Démarrer l'enregistrement continu
        audio_recorder.start_continuous_recording()
        time.sleep(2)  # Laisser le buffer se remplir
        
        # Boucle de transcription en parallèle
        while True:
            try:
                # Récupérer un segment du buffer continu
                print("📊 Analyse segment...", end="", flush=True)
                temp_audio = audio_recorder.get_audio_segment(duration=10)
                
                if temp_audio:
                    # Transcrire
                    print(" 🔄 Transcription...", end="", flush=True)
                    text = transcription_client.transcribe_audio(temp_audio)
                    
                    # Sauvegarder si non vide
                    if text:
                        transcription_client.save_transcription(text)
                        print(" ✅")
                    else:
                        print(" 🔇 (silence)")
                else:
                    print(" ⚠️ (pas de données)")
                
                # Délai avant le prochain segment
                time.sleep(8)  # Analyse toutes les 8 secondes
                
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
