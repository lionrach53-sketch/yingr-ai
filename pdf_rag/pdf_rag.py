import pdfplumber
import json
import os
import re
from tqdm import tqdm
from uuid import uuid4

# ===============================
# CONFIG
# ===============================
PDF_DIR = "pdf_sources"
OUTPUT_FILE = "output/connaissances_pdf_enrichies.json"
LANGUES = ["fr", "mo", "di"]

CHUNK_MIN = 30   # mots (réduit pour générer plus de blocs)
CHUNK_MAX = 350

CATEGORIE_PAR_DEFAUT = "general"
NIVEAU_PAR_DEFAUT = "grand_public"

# ===============================
# UTILS
# ===============================

def nettoyer_texte(text):
    if not text:
        return ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.I)
    return text.strip()

def decouper_en_chunks(texte):
    mots = texte.split()
    chunks = []
    start = 0

    while start < len(mots):
        end = min(start + CHUNK_MAX, len(mots))
        chunk = mots[start:end]

        print(f"[DEBUG] start={start}, end={end}, total={len(mots)}")

        if len(chunk) >= CHUNK_MIN:
            chunks.append(" ".join(chunk))

        next_start = end - 30  # overlap
        if next_start <= start:
            next_start = start + 1  # sécurité pour éviter boucle infinie
        start = next_start
        if start < 0:
            start = 0

    return chunks

def detecter_sous_categorie(texte):
    texte = texte.lower()
    if "mil" in texte:
        return "mil"
    if "maïs" in texte or "mais" in texte:
        return "mais"
    if "riz" in texte:
        return "riz"
    return "general"

# ===============================
# EXTRACTION PDF
# ===============================

def extraire_pdf(path_pdf):
    texte_total = []
    with pdfplumber.open(path_pdf) as pdf:
        for page in pdf.pages:
            texte = page.extract_text()
            if texte:
                texte_total.append(nettoyer_texte(texte))
    return " ".join(texte_total)

# ===============================
# PIPELINE PRINCIPAL
# ===============================

def traiter_pdf(pdf_path):
    print(f"📄 Traitement : {pdf_path}")
    texte = extraire_pdf(pdf_path)

    if not texte:
        print("⚠️ Aucun texte extrait")
        return []

    chunks = decouper_en_chunks(texte)
    connaissances = []

    for chunk in chunks:
        sous_cat = detecter_sous_categorie(chunk)

        connaissance = {
            "id": str(uuid4()),
            "categorie": CATEGORIE_PAR_DEFAUT,
            "sous_categorie": sous_cat,
            "niveau": NIVEAU_PAR_DEFAUT,
            "source": {
                "type": "pdf",
                "fichier": os.path.basename(pdf_path)
            },
            "langues": {
                "fr": {
                    "intention": "information",
                    "question_type": "explication",
                    "reponse": chunk
                },
                "mo": {},
                "di": {}
            }
        }

        connaissances.append(connaissance)

    return connaissances

# ===============================
# MAIN
# ===============================

def main():
    os.makedirs("output", exist_ok=True)

    toutes_connaissances = []

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdfs:
        print("❌ Aucun PDF trouvé")
        return

    for pdf in tqdm(pdfs):
        chemin = os.path.join(PDF_DIR, pdf)
        toutes_connaissances.extend(traiter_pdf(chemin))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(toutes_connaissances, f, ensure_ascii=False, indent=2)

    print("")
    print("✅ TERMINÉ")
    print(f"📦 {len(toutes_connaissances)} chunks générés")
    print(f"📁 Fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
