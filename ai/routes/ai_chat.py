# ai/routes/ai_chat.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from datetime import datetime
import logging

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

        # Gérer les salutations
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
        
        # Pour les questions normales, utiliser le contexte comme réponse
        context_blocks = _rag_context_to_blocks(context)
        
        if context_blocks and len(context_blocks) > 0:
            # Prendre le premier bloc de contexte comme réponse principale
            main_response = _fix_mojibake(context_blocks[0])
            # Garder les autres blocs comme contexte supplémentaire
            additional_context = [_fix_mojibake(b) for b in context_blocks[1:]] if len(context_blocks) > 1 else []
        else:
            # Fallback sur la réponse générée par RAG
            main_response = _fix_mojibake(answer_raw) if answer_raw else "Je n'ai pas trouvé d'information pertinente."
            additional_context = []

        # Récupérer l'historique
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

        # Sauvegarder dans MongoDB
        conversation_data = {
            "user_id": user.get("id"),
            "session_id": session_id,
            "category": req.category,
            "question": req.message,
            "answer": main_response,
            "context": additional_context,
            "language": detected_language,
            "intent": intent,
            "metadata": {"method": "rag_context"},
            "timestamp": datetime.utcnow()
        }
        conversation_id = db.save_chat_conversation(conversation_data)

        return {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "question": req.message,
            "answer": main_response,
            "language": detected_language,
            "intent": intent,
            "context": additional_context,
            "metadata": {"method": "rag_context"}
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
        detected_language = req.language or "fr"
        intent = conversation_service.detect_intent(req.message, detected_language)

        # Salut / Thanks - utiliser conversation_service
        if intent == "greeting":
            greeting_response = conversation_service.generate_greeting_response(detected_language)
            return {
                "session_id": session_id,
                "conversation_id": None,
                "response": greeting_response,
                "language": detected_language,
                "intent": intent,
                "context": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        if intent == "thanks":
            thanks_response = conversation_service.generate_thanks_response(detected_language)
            return {
                "session_id": session_id,
                "conversation_id": None,
                "response": thanks_response,
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

        # RAG avec contexte
        answer_raw, context = rag.ask(
            query=req.message, 
            k=5,
            language=detected_language,
            category=req.category,
            min_confidence=0.40
        )
        
        # Traiter le contexte
        context_blocks = _rag_context_to_blocks(context)
        
        if context_blocks and len(context_blocks) > 0:
            # Prendre le premier bloc comme réponse principale
            main_response = _fix_mojibake(context_blocks[0])
            # Garder les autres comme contexte
            additional_context = [_fix_mojibake(b) for b in context_blocks[1:]] if len(context_blocks) > 1 else []
        else:
            main_response = _fix_mojibake(answer_raw) if answer_raw else "Je n'ai pas trouvé d'information pertinente."
            additional_context = []

        return {
            "session_id": session_id,
            "conversation_id": None,
            "response": main_response,
            "language": detected_language,
            "intent": intent,
            "context": additional_context,
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
        
        # Détecter l'intention
        intent = conversation_service.detect_intent(req.message, detected_language)
        
        # Gérer les salutations avec conversation_service
        if intent == "greeting":
            greeting_response = conversation_service.generate_greeting_response(detected_language)
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", greeting_response)
            
            return {
                "session_id": session_id,
                "response": greeting_response,
                "language": detected_language,
                "intent": intent,
                "mode": "greeting",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if intent == "thanks":
            thanks_response = conversation_service.generate_thanks_response(detected_language)
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", thanks_response)
            
            return {
                "session_id": session_id,
                "response": thanks_response,
                "language": detected_language,
                "intent": intent,
                "mode": "thanks",
                "timestamp": datetime.utcnow().isoformat()
            }

        # RAG enrichi
        understanding = QueryUnderstanding.understand_health_query(req.message)
        expanded_query = understanding['reformulated_query'] if understanding else req.message
        answer_raw, context_raw = rag.ask(
            query=expanded_query,
            k=10,
            language=detected_language,
            category=req.category,
            min_confidence=0.15
        )
        
        # Traiter le contexte RAG
        context_blocks = _rag_context_to_blocks(context_raw)
        
        # Déterminer la réponse principale
        if context_blocks and len(context_blocks) > 0:
            # Prendre le premier bloc de contexte comme réponse principale
            main_response = _fix_mojibake(context_blocks[0])
            # Garder les autres blocs comme contexte supplémentaire
            additional_context = [_fix_mojibake(b) for b in context_blocks[1:]] if len(context_blocks) > 1 else []
            
            # Utiliser AI Brain pour améliorer la réponse si nécessaire
            rag_results = [{"question": req.message, "reponse": b} for b in context_blocks[:3]]
            intelligent_response = ai_brain.generate_intelligent_response(
                question=req.message,
                rag_results=rag_results,
                category=req.category,
                language=detected_language
            )
            
            # Combiner réponse RAG avec amélioration IA
            enhanced_response = intelligent_response["reponse"] if intelligent_response else main_response
            mode = intelligent_response.get("mode", "rag_enhanced") if intelligent_response else "rag_context"
        else:
            # Fallback sur la réponse générée par AI Brain
            enhanced_response = "Je n'ai pas trouvé d'information pertinente dans ma base de connaissances."
            additional_context = []
            mode = "fallback"

        # Audio pour mooré et dioula
        audio_url, audio_mode = None, "not_available"
        if detected_language in ["mo", "di"]:
            try:
                audio_url, audio_mode = tts_service.generate_audio(
                    text=enhanced_response,
                    language=detected_language
                )
            except Exception as e:
                logger.warning(f"⚠️ Audio non disponible: {e}")

        # Construire le payload
        payload = {
            "session_id": session_id,
            "response": enhanced_response,
            "language": detected_language,
            "intent": intent,
            "category": req.category,
            "sources_count": len(context_blocks),
            "mode": mode,
            "context": additional_context,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_url": audio_url,
            "audio_mode": audio_mode
        }

        # Save conversation
        try:
            if db:
                conversation_data = {
                    "user_id": f"guest_{session_id}",
                    "session_id": session_id,
                    "category": payload["category"],
                    "question": req.message,
                    "answer": enhanced_response,
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