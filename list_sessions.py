#!/usr/bin/env python3
"""
Script pour lister toutes les sessions de transcription
"""
import os
import json
from datetime import datetime

def list_transcription_sessions():
    """Liste toutes les sessions de transcription"""
    if not os.path.exists("transcription"):
        print("❌ Aucun dossier transcription/ trouvé")
        return
    
    files = [f for f in os.listdir("transcription") if f.startswith("session_") and f.endswith(".json")]
    
    if not files:
        print("❌ Aucune session trouvée")
        return
    
    print(f"📁 {len(files)} session(s) trouvée(s) :\n")
    
    for file in sorted(files):
        filepath = f"transcription/{file}"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = data.get('session', {})
            transcriptions = data.get('transcriptions', [])
            
            start_time = session.get('start_time', 'N/A')
            status = session.get('status', 'unknown')
            model = session.get('model', 'N/A')
            count = len(transcriptions)
            
            print(f"📄 {file}")
            print(f"   📅 Début: {start_time[:19] if start_time != 'N/A' else 'N/A'}")
            print(f"   📊 Status: {status}")
            print(f"   🤖 Modèle: {model}")
            print(f"   💬 Transcriptions: {count}")
            print()
            
        except Exception as e:
            print(f"❌ Erreur lecture {file}: {e}")

if __name__ == "__main__":
    list_transcription_sessions()