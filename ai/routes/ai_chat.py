# ai/routes/ai_chat.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from datetime import datetime
import logging
import random

from ..service.rag import RAGService
from ..service.conversation import ConversationService
from ..service.ai_brain import ai_brain
from ..service.text_normalizer import text_normalizer
from ..service.tts_service import tts_service
from ..service.stt_service import stt_service
from ..service.query_understanding import QueryUnderstanding

logger = logging.getLogger(__name__)

try:
    from mongodb import db
except ImportError:
    from backend.mongodb import db

try:
    from ...security import require_expert
except ImportError:
    from security import require_expert

router = APIRouter(prefix="/ai", tags=["YINGRE AI"])

# ==============================
# Helpers
# ==============================
def _fix_mojibake(text: str) -> str:
    """Tente de corriger un texte mal décodé (ex: 'Ã©' -> 'é')."""
    if not text or not isinstance(text, str):
        return text
    if "Ã" not in text:
        return text
    try:
        return text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text

def _rag_context_to_blocks(context_raw):
    """Normalise le contexte retourné par RAGService.ask() en liste de blocs."""
    if not context_raw:
        return []
    if isinstance(context_raw, list):
        blocks = []
        for item in context_raw:
            if not item:
                continue
            if isinstance(item, str):
                blocks.extend([b.strip() for b in item.split("\n\n---\n\n") if b.strip()])
            else:
                blocks.append(str(item).strip())
        return [b for b in blocks if b]
    if isinstance(context_raw, str):
        return [b.strip() for b in context_raw.split("\n\n---\n\n") if b.strip()]
    return [str(context_raw).strip()]

# ==============================
# Initialiser les services
# ==============================
rag = RAGService()
conversation_service = ConversationService()

# ==============================
# Request Models
# ==============================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    category: Optional[str] = "general"
    language: Optional[str] = "fr"

# ==============================
# Routes
# ==============================
@router.post("/chat")
def chat(req: ChatRequest, user=Depends(require_expert)):
    try:
        session_id = req.session_id or str(uuid4())
        detected_language = req.language or "fr"
        intent = conversation_service.detect_intent(req.message, detected_language)

        if intent == 'greeting':
            greeting_response = conversation_service.generate_greeting_response(detected_language)
            return {
                "session_id": session_id,
                "conversation_id": None,
                "question": req.message,
                "answer": greeting_response,
                "language": detected_language,
                "intent": intent,
                "context": [],
                "metadata": {"method": "greeting"}
            }

        if intent == 'thanks':
            thanks_response = conversation_service.generate_thanks_response(detected_language)
            return {
                "session_id": session_id,
                "conversation_id": None,
                "question": req.message,
                "answer": thanks_response,
                "language": detected_language,
                "intent": intent,
                "context": [],
                "metadata": {"method": "thanks"}
            }

        # Vérifier si clarification nécessaire
        should_clarify, clarification = ai_brain.should_ask_clarification(req.message, detected_language)
        if should_clarify:
            return {
                "session_id": session_id,
                "conversation_id": None,
                "question": req.message,
                "answer": clarification,
                "language": detected_language,
                "intent": "clarification_needed",
                "context": [],
                "metadata": {"method": "clarification"}
            }

        # Contexte RAG
        answer_raw, context = rag.ask(req.message, k=5)
        history = []
        try:
            past_conversations = db.get_chat_conversations(user_id=user.get("id"))
            history = [
                {"question": conv.get("question"), "answer": conv.get("answer")}
                for conv in past_conversations
                if conv.get("session_id") == session_id
            ][-3:]
        except:
            history = []

        # Réponse intelligente
        rag_context_full = context if isinstance(context, str) else ("\n\n".join(context) if context else "")
        intelligent_answer, metadata = ai_brain.generate_intelligent_response(
            question=req.message,
            rag_context=rag_context_full,
            language=detected_language,
            conversation_history=history
        )

        conversation_data = {
            "user_id": user.get("id"),
            "session_id": session_id,
            "category": req.category,
            "question": req.message,
            "answer": intelligent_answer,
            "context": context,
            "language": detected_language,
            "intent": intent,
            "metadata": metadata,
            "timestamp": datetime.utcnow()
        }
        conversation_id = db.save_chat_conversation(conversation_data)

        return {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "question": req.message,
            "answer": intelligent_answer,
            "language": detected_language,
            "intent": intent,
            "context": context,
            "metadata": metadata
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur du service AI: {str(e)}")


@router.get("/history")
def get_history(user=Depends(require_expert), session_id: Optional[str] = None, limit: int = 50):
    try:
        query = {"user_id": user.get("id")}
        if session_id:
            query["session_id"] = session_id
        conversations = db.get_chat_conversations(user_id=user.get("id"))
        return conversations[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération historique: {str(e)}")


@router.post("/chat/guest")
def chat_guest(req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid4())
        detected_language = conversation_service.detect_language(req.message)
        intent = conversation_service.detect_intent(req.message, detected_language)

        # Salut / Thanks
        if intent == "greeting":
            greetings = {
                "fr": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "mo": "Kɩbare ! Tõnd nonglem maana yaa ?",
                "di": "I ni sɔgɔma ! N bɛ se ka i dɛmɛ di cogo jumɛn na ?"
            }
            return {
                "session_id": session_id,
                "conversation_id": None,
                "response": greetings.get(detected_language, greetings["fr"]),
                "language": detected_language,
                "intent": intent,
                "context": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        if intent == "thanks":
            thanks_responses = {
                "fr": "Je vous en prie ! N'hésitez pas si vous avez d'autres questions.",
                "mo": "Barka ! Kãadem b sã yɩɩ n kɩt yõodo.",
                "di": "Baaraka ! Aw bɛna ɲininkali wɛrɛw kɛ wa, i k'a fɔ."
            }
            return {
                "session_id": session_id,
                "conversation_id": None,
                "response": thanks_responses.get(detected_language, thanks_responses["fr"]),
                "language": detected_language,
                "intent": intent,
                "context": [],
                "timestamp": datetime.utcnow().isoformat()
            }

        # Clarification
        should_clarify, clarification = ai_brain.should_ask_clarification(req.message, detected_language)
        if should_clarify:
            return {
                "session_id": session_id,
                "conversation_id": None,
                "response": clarification,
                "language": detected_language,
                "intent": "clarification_needed",
                "context": [],
                "metadata": {"method": "clarification"},
                "timestamp": datetime.utcnow().isoformat()
            }

        # RAG
        answer_raw, context = rag.ask(
            query=req.message, 
            k=5,
            language=detected_language,
            category=req.category,
            min_confidence=0.40
        )

        return {
            "session_id": session_id,
            "conversation_id": None,
            "response": answer_raw,
            "language": detected_language,
            "intent": intent,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur du service AI: {str(e)}")


# -----------------------------
# chat_intelligent & chat_voice
# -----------------------------


@router.post("/chat/intelligent")
def chat_intelligent(req: ChatRequest):
    try:
        original_message = req.message
        normalized_message = text_normalizer.normalize(req.message)
        req.message = normalized_message
        session_id = req.session_id or str(uuid4())
        detected_language = req.language or "fr"
        intent = conversation_service.detect_intent(req.message, detected_language)

        # RAG enrichi avec paramètres OPTIMISÉS
        understanding = QueryUnderstanding.understand_health_query(req.message)
        expanded_query = understanding['reformulated_query'] if understanding else req.message
        
        # 🔥 PARAMÈTRES OPTIMISÉS POUR 8GB RAM
        answer_raw, context_raw = rag.ask(
            query=expanded_query,
            k=3,  # Réduit de 10 à 3 pour moins de tokens
            language=detected_language,
            category=req.category,
            min_confidence=0.65  # Augmenté pour plus de pertinence
        )
        
        # 🔥 GESTION AMÉLIORÉE DU CONTEXTE
        # Si le RAG retourne une réponse de fallback, on l'utilise directement
        fallback_phrases = [
            "Je n'ai pas d'information",
            "Je n'ai pas cette information",
            "M pa tara tagmasg",
            "N tɛ kunnafoni"
        ]
        
        is_fallback = any(phrase in answer_raw for phrase in fallback_phrases)
        
        if is_fallback:
            # C'est une réponse de fallback, on la retourne telle quelle
            main_response = answer_raw
            additional_context = []
        else:
            # C'est une réponse normale, on traite avec ai_brain
            rag_results = [{"question": req.message, "reponse": b} for b in _rag_context_to_blocks(context_raw)[:2]]  # 2 max
            intelligent_response = ai_brain.generate_intelligent_response(
                question=req.message,
                rag_results=rag_results,
                category=req.category,
                language=detected_language
            )
            
            context_blocks = _rag_context_to_blocks(context_raw)
            if context_blocks and len(context_blocks) > 0:
                main_response = _fix_mojibake(context_blocks[0])
                additional_context = [_fix_mojibake(b) for b in context_blocks[1:]] if len(context_blocks) > 1 else []
            else:
                main_response = _fix_mojibake(intelligent_response["reponse"])
                additional_context = []

        # Audio (inchangé)
        audio_url, audio_mode = None, "not_available"
        if detected_language in ["mo", "di"]:
            try:
                audio_url, audio_mode = tts_service.generate_audio(
                    text=main_response,
                    language=detected_language
                )
            except Exception as e:
                logger.warning(f"⚠️ Audio non disponible: {e}")

        # Payload final
        payload = {
            "session_id": session_id,
            "response": main_response,
            "language": detected_language,
            "intent": intent,
            "category": req.category if not is_fallback else "general",
            "sources_count": 0 if is_fallback else len(additional_context) + 1,
            "mode": "fallback" if is_fallback else intelligent_response.get("mode", "intelligent"),
            "context": additional_context,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_url": audio_url,
            "audio_mode": audio_mode,
            "is_fallback": is_fallback  # Nouveau champ pour debug
        }

        # Save conversation (inchangé)
        try:
            if db:
                conversation_data = {
                    "user_id": f"guest_{session_id}",
                    "session_id": session_id,
                    "category": payload["category"],
                    "question": req.message,
                    "answer": main_response,
                    "context": payload["context"],
                    "language": detected_language,
                    "intent": intent,
                    "mode": payload["mode"],
                    "audio_url": audio_url,
                    "audio_mode": audio_mode,
                    "timestamp": datetime.utcnow(),
                }
                db.save_chat_conversation(conversation_data)
        except Exception as e:
            logger.warning(f"⚠️ Impossible de sauvegarder la conversation: {e}")

        return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur service AI intelligent: {str(e)}")



@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None,
    category: Optional[str] = "general",
    language: Optional[str] = "fr"
):
    try:
        if not stt_service.is_available():
            raise HTTPException(status_code=503, detail="Service de reconnaissance vocale non disponible.")
        audio_bytes = await audio.read()
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Fichier audio trop court ou invalide")
        transcription, detected_language, confidence = stt_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=audio.filename,
            language=language
        )

        # Appel interne à chat_intelligent
        req_chat = ChatRequest(
            message=transcription,
            session_id=session_id,
            category=category,
            language=language
        )
        intelligent_result = chat_intelligent(req_chat)
        intelligent_result["transcription"] = transcription
        intelligent_result["transcription_confidence"] = confidence
        intelligent_result["stt_service"] = "whisper"
        intelligent_result["workflow"] = "voice → stt → rag+llm → tts → voice"
        return intelligent_result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur service chat vocal: {str(e)}")


@router.post("/chat/clear-history")
def clear_chat_history():
    try:
        ai_brain.clear_history()
        return {"status": "ok", "message": "Historique effacé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))