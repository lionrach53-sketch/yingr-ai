import csv
import json
from pathlib import Path
from datetime import datetime

"""
Script utilitaire pour préparer un corpus voix + RAG à partir de tes fichiers MP3.

Entrées attendues :
- Un dossier contenant tes fichiers audio MP3 (par ex. backend/audio/raw)
- Un fichier CSV de métadonnées décrivant chaque audio, par ex. corpus_audio.csv

Format CSV attendu (séparateur virgule) :

    filename,language,domain,text
    audio_001.mp3,fr,agriculture,"Texte exact de la voix off..."
    audio_002.mp3,mo,agriculture,"..."
    audio_003.mp3,di,finance,"..."

- filename : nom du fichier MP3 (doit exister dans le dossier audio)
- language : fr | mo | di
- domain   : agriculture | finance | transformation | autre
- text     : texte de la voix off (transcription ou script)

Sorties :
1) backend/audio/metadata_stt.csv
   -> corpus pour tester/évaluer Whisper (STT)

2) backend/ingest/connaissances_enrichies_from_audio_<timestamp>.json
   -> squelette prêt pour ingestion RAG au format enrichi de YINGR-AI
"""

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio"  # tu peux ajuster

# Dossiers audio par langue
# - mo (mooré)   -> backend/audio/moree
# - di (dioula)  -> backend/audio/dioula
# - fr (français)-> par défaut backend/audio/raw (à créer si nécessaire)
LANGUAGE_AUDIO_DIRS = {
    "mo": AUDIO_DIR / "moree",
    "di": AUDIO_DIR / "dioula",
}

# Dossier par défaut si aucune correspondance de langue
DEFAULT_AUDIO_DIR = AUDIO_DIR / "raw"  # tu peux créer ce dossier pour les autres cas

CORPUS_CSV = AUDIO_DIR / "corpus_audio.csv"
INGEST_DIR = BASE_DIR / "ingest"


def prepare_stt_corpus():
    """Génère metadata_stt.csv pour le corpus de reconnaissance vocale."""
    input_csv = CORPUS_CSV
    output_csv = AUDIO_DIR / "metadata_stt.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable: {input_csv}")

    rows = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            filename = row.get("filename", "").strip()
            language = row.get("language", "").strip().lower() or "fr"
            domain = row.get("domain", "").strip().lower() or "general"
            text = row.get("text", "").strip()

            if not filename or not text:
                # On ignore les lignes incomplètes
                continue

            # On choisit le dossier en fonction de la langue si possible,
            # sinon on bascule sur le dossier par défaut (raw).
            audio_base_dir = LANGUAGE_AUDIO_DIRS.get(language, DEFAULT_AUDIO_DIR)
            audio_path = audio_base_dir / filename
            if not audio_path.exists():
                print(f"⚠️ Audio manquant pour la ligne {i}: {audio_path}")
                continue

            rows.append({
                "audio_path": str(audio_path.relative_to(BASE_DIR)),
                "text": text,
                "language": language,
                "domain": domain,
            })

    if not rows:
        print("⚠️ Aucune ligne valide trouvée pour le corpus STT.")
        return

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["audio_path", "text", "language", "domain"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Corpus STT généré: {output_csv} ({len(rows)} entrées)")


def prepare_rag_corpus():
    """Génère un JSON de connaissances enrichies à partir des textes de voix off."""
    input_csv = CORPUS_CSV
    if not input_csv.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable: {input_csv}")

    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = INGEST_DIR / f"connaissances_enrichies_from_audio_{timestamp}.json"

    entries = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            filename = row.get("filename", "").strip()
            language = row.get("language", "").strip().lower() or "fr"
            domain = row.get("domain", "").strip().lower() or "general"
            text = row.get("text", "").strip()

            if not text:
                continue

            # On construit une brique de connaissance très simple par défaut.
            # Tu pourras affiner plus tard (reponse_courte, conseil, avertissement...).
            base_entry = {
                "categorie": domain,
                "sous_categorie": "audio_corpus",
                "niveau": "base",
                "source_audio": filename,
            }

            # Champs par langue
            if language == "fr":
                base_entry.update({
                    "intention_fr": "enseignement_oral",
                    "question_type_fr": "explication",
                    "reponse_courte_fr": text[:160] + ("..." if len(text) > 160 else ""),
                    "reponse_detaillee_fr": text,
                    "conseil_fr": "",
                    "avertissement_fr": "",
                })
            elif language == "mo":
                base_entry.update({
                    "intention_mo": "enseignement_oral",
                    "question_type_mo": "explication",
                    "reponse_courte_mo": text[:160] + ("..." if len(text) > 160 else ""),
                    "reponse_detaillee_mo": text,
                    "conseil_mo": "",
                    "avertissement_mo": "",
                })
            elif language == "di":
                base_entry.update({
                    "intention_di": "enseignement_oral",
                    "question_type_di": "explication",
                    "reponse_courte_di": text[:160] + ("..." if len(text) > 160 else ""),
                    "reponse_detaillee_di": text,
                    "conseil_di": "",
                    "avertissement_di": "",
                })
            else:
                # Par défaut, on range tout en français si langue inconnue
                base_entry.update({
                    "intention_fr": "enseignement_oral",
                    "question_type_fr": "explication",
                    "reponse_courte_fr": text[:160] + ("..." if len(text) > 160 else ""),
                    "reponse_detaillee_fr": text,
                    "conseil_fr": "",
                    "avertissement_fr": "",
                })

            entries.append(base_entry)

    if not entries:
        print("⚠️ Aucune entrée générée pour le RAG.")
        return

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"✅ Corpus RAG généré: {output_json} ({len(entries)} entrées)")


if __name__ == "__main__":
    print("📂 Base backend:", BASE_DIR)
    print("🎧 Dossier audio par défaut:", DEFAULT_AUDIO_DIR)
    print("🎧 Dossier mooré (mo):", LANGUAGE_AUDIO_DIRS.get("mo"))
    print("🎧 Dossier dioula (di):", LANGUAGE_AUDIO_DIRS.get("di"))
    print("📄 Métadonnées CSV:", CORPUS_CSV)

    prepare_stt_corpus()
    prepare_rag_corpus()
