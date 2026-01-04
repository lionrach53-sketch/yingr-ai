# ai/routes/ai_ingest.py
print("[DEBUG] >>> ai_ingest.py chargé")

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List
import logging
from datetime import datetime
import uuid
import shutil
import os

from ..service.rag import RAGService
try:
    from mongodb import db  # type: ignore
except ImportError:
    from backend.mongodb import db  # type: ignore

try:
    from ...security import require_admin_or_expert
except ImportError:
    from security import require_admin_or_expert

logger = logging.getLogger(__name__)

logger.info("[YINGRE AI] Initialisation du RAGService pour l'ingest...")
rag = RAGService()
logger.info("[YINGRE AI] RAGService prêt pour l'ingest.")

security_scheme = HTTPBearer()

router = APIRouter(
    prefix="/ai",
    tags=["YINGRE AI"],
    dependencies=[Depends(security_scheme)]
)

# -------------------------------
# Models
# -------------------------------
class IngestRequest(BaseModel):
    texts: List[str]
    source: str = "unknown"

# -------------------------------
# Endpoint pour ingérer du texte
# -------------------------------
@router.post("/ingest")
def ingest_text(req: IngestRequest, user=Depends(require_admin_or_expert)):
    try:
        # Ajouter chaque texte dans RAG + MongoDB documents
        for t in req.texts:
            rag.ingest([t], req.source)  # RAG
            db.add_document({
                "content": t,
                "source": req.source,
                "uploaded_by": user.get("id", "anonymous"),
                "uploaded_at": datetime.utcnow(),
                "type": "text"
            })
        return {"status": "ok", "ingested_count": len(req.texts)}
    except Exception as e:
        logger.error(f"❌ Erreur ingestion texte: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Endpoint pour ingérer des images / photos
# -------------------------------
@router.post("/ingest/photo")
def ingest_photo(
    file: UploadFile = File(...),
    source: str = Form("unknown"),
    user=Depends(require_admin_or_expert)
):
    try:
        # 1️⃣ Sauvegarde du fichier temporaire
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_ext = os.path.splitext(file.filename)[1]
        saved_path = os.path.join(upload_dir, f"{uuid.uuid4()}{file_ext}")
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2️⃣ Extraction texte si possible (OCR)
        try:
            from pytesseract import image_to_string  # type: ignore
            from PIL import Image
            text_content = image_to_string(Image.open(saved_path))
        except ImportError:
            text_content = ""
            logger.warning("❌ pytesseract non installé, texte non extrait")

        # 3️⃣ Ajouter texte à RAG si disponible
        if text_content.strip():
            rag.ingest([text_content], source)

        # 4️⃣ Enregistrer dans MongoDB
        doc_id = db.add_document({
            "file_path": saved_path,
            "source": source,
            "uploaded_by": user.get("id", "anonymous"),
            "uploaded_at": datetime.utcnow(),
            "type": "image",
            "extracted_text": text_content
        })

        return {"status": "ok", "document_id": doc_id, "extracted_text": text_content}

    except Exception as e:
        logger.error(f"❌ Erreur ingestion photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
