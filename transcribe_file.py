#!/usr/bin/env python3
"""
Script pour transcrire un fichier audio existant
Usage: python transcribe_file.py chemin/vers/fichier.mp3
"""

import sys
import os
import json
import ssl
from datetime import datetime
import whisper

# Fix SSL pour macOS
ssl._create_default_https_context = ssl._create_unverified_context

class FileTranscriptionClient:
    """Client pour transcrire des fichiers audio"""
    
    def __init__(self, model="large", lang="fr"):
        self.model_name = model
        self.lang = lang
        
        # Créer le dossier transcription
        os.makedirs("transcription", exist_ok=True)
        
        # Charger le modèle Whisper
        print(f"🤖 Chargement du modèle Whisper '{model}'...")
        self.whisper_model = whisper.load_model(model)
        print("✅ Modèle chargé!")
    
    def transcribe_file(self, audio_file):
        """Transcrit un fichier audio complet"""
        if not os.path.exists(audio_file):
            print(f"❌ Fichier non trouvé: {audio_file}")
            return None
        
        print(f"🎵 Analyse du fichier: {audio_file}")
        
        try:
            # Paramètres optimaux pour cours/amphithéâtre
            result = self.whisper_model.transcribe(
                audio_file,
                language=self.lang,
                fp16=False,
                logprob_threshold=-1.0,
                temperature=0.0,
                compression_ratio_threshold=1.35,
                condition_on_previous_text=True,
                word_timestamps=True,  # Timestamps utiles pour fichiers longs
                no_speech_threshold=0.2
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            return None
    
    def save_transcription(self, result, audio_file):
        """Sauvegarde la transcription complète"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        json_file = f"transcription/{base_name}_{timestamp}.json"
        
        # Données de session
        session_data = {
            "session": {
                "timestamp": datetime.now().isoformat(),
                "source_file": audio_file,
                "model": self.model_name,
                "language": self.lang,
                "duration": result.get("segments", [])[-1].get("end", 0) if result.get("segments") else 0
            },
            "transcription": {
                "full_text": result["text"],
                "segments": []
            }
        }
        
        # Segments avec timestamps
        for segment in result.get("segments", []):
            session_data["transcription"]["segments"].append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        # Sauvegarder JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarder aussi en TXT simple
        txt_file = f"transcription/{base_name}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        print(f"📄 Transcription sauvée:")
        print(f"   JSON: {json_file}")
        print(f"   TXT:  {txt_file}")
        
        return json_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe_file.py <fichier_audio>")
        print("Formats supportés: mp3, wav, m4a, ogg, etc.")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("🎯 Transcription de fichier audio")
    print("=" * 50)
    
    try:
        # Créer le client
        client = FileTranscriptionClient(model="large", lang="fr")
        
        # Transcrire
        print("\n🔄 Transcription en cours...")
        result = client.transcribe_file(audio_file)
        
        if result:
            # Sauvegarder
            json_file = client.save_transcription(result, audio_file)
            
            # Statistiques
            duration = result.get("segments", [])[-1].get("end", 0) if result.get("segments") else 0
            segments = len(result.get("segments", []))
            
            print(f"\n✅ Transcription terminée!")
            print(f"⏱️  Durée audio: {duration:.1f}s")
            print(f"📊 Segments: {segments}")
            print(f"📝 Longueur: {len(result['text'])} caractères")
            print(f"\n📄 Texte complet:")
            print("-" * 30)
            print(result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"])
        else:
            print("❌ Échec de la transcription")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Transcription interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()