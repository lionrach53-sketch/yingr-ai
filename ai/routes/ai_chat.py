from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Services existants
from ..service.ai_brain import ai_brain
from ..service.rag_engine import rag_engine
from ..service.stt_service import speech_to_text
from ..service.tts_service import text_to_speech
from ..service.language_detector import detect_language

# ===============================
# ✅ CORRECTION CRITIQUE
# ===============================
# intelligent_chat était utilisé mais jamais défini
# On le mappe proprement vers ai_brain
intelligent_chat = ai_brain

router = APIRouter()

# ===============================
# 📦 SCHEMAS
# ===============================

class ChatRequest(BaseModel):
    message: Optional[str] = None
    language: Optional[str] = "fr"
    use_rag: Optional[bool] = True
    audio_base64: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    language: str
    used_rag: bool
    audio_base64: Optional[str] = None


# ===============================
# 🧠 ROUTE CHAT SIMPLE
# ===============================

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if not request.message and not request.audio_base64:
        raise HTTPException(status_code=400, detail="Message ou audio requis")

    # 🎙️ Speech to Text
    message = request.message
    if request.audio_base64:
        message = speech_to_text(request.audio_base64)

    # 🌍 Détection de langue si absente
    language = request.language or detect_language(message)

    # 📚 RAG
    used_rag = False
    context = ""
    if request.use_rag:
        context = rag_engine.search(message)
        used_rag = True

    # 🧠 Génération réponse
    response_text = ai_brain.generate_response(
        prompt=message,
        context=context,
        language=language
    )

    # 🔊 Text to Speech
    audio_response = text_to_speech(response_text, language)

    return ChatResponse(
        response=response_text,
        language=language,
        used_rag=used_rag,
        audio_base64=audio_response
    )


# ===============================
# 🤖 ROUTE CHAT INTELLIGENT
# ===============================

@router.post("/chat/intelligent", response_model=ChatResponse)
def intelligent_chat_route(request: ChatRequest):

    if not request.message and not request.audio_base64:
        raise HTTPException(status_code=400, detail="Message ou audio requis")

    # 🎙️ STT
    message = request.message
    if request.audio_base64:
        message = speech_to_text(request.audio_base64)

    # 🌍 Langue
    language = request.language or detect_language(message)

    # 🧠 Analyse intelligente (clarification, intention, etc.)
    if intelligent_chat.should_ask_clarification(message):
        clarification = intelligent_chat.ask_clarification(message, language)

        return ChatResponse(
            response=clarification,
            language=language,
            used_rag=False,
            audio_base64=text_to_speech(clarification, language)
        )

    # 📚 RAG avancé
    used_rag = False
    context = ""
    if request.use_rag:
        context = rag_engine.search(message)
        used_rag = True

    # 🤖 Réponse intelligente
    response_text = intelligent_chat.generate_intelligent_response(
        user_input=message,
        context=context,
        language=language
    )

    # 🔊 TTS
    audio_response = text_to_speech(response_text, language)

    return ChatResponse(
        response=response_text,
        language=language,
        used_rag=used_rag,
        audio_base64=audio_response
    )


# ===============================
# 🩺 ROUTE HEALTH CHECK
# ===============================

@router.get("/health")
def health():
    return {
        "status": "ok",
        "ai": "YINGRE AI",
        "rag": "enabled",
        "intelligent_chat": True
    }
