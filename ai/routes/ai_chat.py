from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from ..service.rag import rag_service
from ..service.ai_brain import ai_brain
from ..service.intelligent_chat import intelligent_chat
from ..utils.text_utils import detect_language, fix_mojibake

logger = logging.getLogger(__name__)
router = APIRouter()

# =========================
# 📦 MODELS
# =========================

class ChatRequest(BaseModel):
    message: str
    category: Optional[str] = None
    language: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source: str


# =========================
# 🧠 CORE FUNCTION
# =========================

def _process_chat(question: str, category: Optional[str], language: Optional[str]):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message vide")

    detected_language = language or detect_language(question)

    logger.info(f"🧠 Question: {question}")
    logger.info(f"🌍 Langue: {detected_language}")

    # 🔍 RAG
    answer_raw, context_raw, rag_results = rag_service.ask(
        question=question,
        category=category,
        language=detected_language
    )

    # 🧠 LLM
    intelligent_response = ai_brain.generate_intelligent_response(
        question=question,
        rag_results=rag_results,
        category=category,
        language=detected_language
    )

    final_answer = intelligent_response.get("reponse")

    # 🔁 FALLBACK LOGIQUE
    if not final_answer or len(final_answer.strip()) < 30:
        logger.warning("⚠️ Réponse LLM faible → fallback RAG")
        if answer_raw and len(answer_raw.strip()) > 30:
            final_answer = answer_raw
            source = "rag"
        else:
            final_answer = (
                "Je n’ai pas encore assez d’informations structurées pour répondre précisément. "
                "Peux-tu préciser ta question ?"
            )
            source = "fallback"
    else:
        source = "rag+llm"

    final_answer = fix_mojibake(final_answer)

    confidence = 0.0
    if rag_results:
        confidence = max(doc.get("score", 0.0) for doc in rag_results)

    return {
        "answer": final_answer,
        "confidence": round(confidence, 3),
        "source": source
    }


# =========================
# 🤖 ROUTES
# =========================

@router.post("/ai/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return _process_chat(req.message, req.category, req.language)


@router.post("/ai/chat/guest", response_model=ChatResponse)
def chat_guest(req: ChatRequest):
    return _process_chat(req.message, req.category, req.language)


@router.post("/ai/chat/intelligent", response_model=ChatResponse)
def chat_intelligent(req: ChatRequest):
    return _process_chat(req.message, req.category, req.language)
