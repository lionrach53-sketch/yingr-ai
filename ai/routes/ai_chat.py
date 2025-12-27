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
    """Normalise le contexte retourné par RAGService.ask() en liste de blocs.

    RAGService.ask() retourne actuellement un contexte texte où les documents sont
    séparés par "\n\n---\n\n" (souvent 'answers-only').
    Cette fonction gère aussi le cas où un ancien code renverrait une liste.
    """
    if not context_raw:
        return []

    # Compatibilité si un ancien code retourne une liste de strings
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

# Initialiser les services
rag = RAGService()
conversation_service = ConversationService()

# ==============================
# Request Models
# ==============================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    category: Optional[str] = "general"  # Catégorie par défaut
    language: Optional[str] = "fr"  # Langue choisie par l'utilisateur (fr, mo, di)


# ==============================
# Routes
# ==============================
@router.post("/chat")
def chat(req: ChatRequest, user=Depends(require_expert)):
    """
    Endpoint pour poser une question à l'IA INTELLIGENTE.
    
    NOUVEAU: L'IA analyse, reformule et raisonne au lieu de copier-coller
    - Détecte la langue automatiquement
    - Reformule intelligemment avec un LLM (LLaMA 2 local)
    - Pose des questions de clarification si nécessaire
    - Guide l'utilisateur pas à pas
    - Sauvegarde dans MongoDB avec session_id
    """
    try:
        # 1️⃣ Déterminer la session
        session_id = req.session_id or str(uuid4())
        
        # 2️⃣ Utiliser la langue choisie par l'utilisateur (pas d'auto-détection)
        detected_language = req.language or "fr"
        
        # 3️⃣ Détecter l'intent (salutation, question, remerciement)
        intent = conversation_service.detect_intent(req.message, detected_language)
        
        # 4️⃣ Gérer les salutations et remerciements
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
        
        # 5️⃣ Vérifier si on doit demander une clarification
        should_clarify, clarification = intelligent_chat.should_ask_clarification(
            req.message, detected_language
        )
        
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

        # 6️⃣ Récupérer le contexte du RAG (multiple documents)
        answer_raw, context = rag.ask(req.message, k=5)  # Top 5 docs pertinents
        
        # 7️⃣ Récupérer l'historique de cette session
        history = []
        try:
            past_conversations = db.get_chat_conversations(user_id=user.get("id"))
            # Filtrer par session_id et prendre les 3 derniers
            history = [
                {"question": conv.get("question"), "answer": conv.get("answer")}
                for conv in past_conversations
                if conv.get("session_id") == session_id
            ][-3:]
        except:
            history = []
        
        # 8️⃣ 🎯 GÉNÉRER UNE RÉPONSE INTELLIGENTE avec le LLM
        # PLUS de copier-coller ! L'IA analyse et reformule
        # RAGService.ask() renvoie un contexte texte (pas une liste)
        rag_context_full = context if isinstance(context, str) else ("\n\n".join(context) if context else "")
        
        intelligent_answer, metadata = intelligent_chat.generate_intelligent_response(
            question=req.message,
            rag_context=rag_context_full,
            language=detected_language,
            conversation_history=history
        )

        # 9️⃣ Sauvegarder dans MongoDB
        conversation_data = {
            "user_id": user.get("id"),
            "session_id": session_id,
            "category": req.category,
            "question": req.message,
            "answer": intelligent_answer,  # Réponse INTELLIGENTE, pas raw
            "context": context,
            "language": detected_language,
            "intent": intent,
            "metadata": metadata,  # Infos sur le LLM utilisé
            "timestamp": datetime.utcnow()
        }

        conversation_id = db.save_chat_conversation(conversation_data)

        # 🔟 Retourner la réponse intelligente
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
    """
    Récupère l'historique des conversations pour un utilisateur ou une session
    """
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
    """
    Endpoint PUBLIC pour utilisateurs invités (sans authentification).
    
    Même intelligence que /ai/chat mais sans besoin d'être expert.
    - Détecte la langue automatiquement
    - Reformule intelligemment avec un LLM (LLaMA 2 local)
    - Pose des questions de clarification si nécessaire
    - Guide l'utilisateur pas à pas
    """
    try:
        # 1️⃣ Déterminer la session
        session_id = req.session_id or str(uuid4())
        
        # 2️⃣ Détecter la langue de la question
        detected_language = conversation_service.detect_language(req.message)
        
        # 3️⃣ Détecter l'intent (salutation, question, remerciement)
        try:
            intent = conversation_service.detect_intent(req.message, detected_language)
            logger.info(f"🎯 Intent détecté: {intent} pour '{req.message[:50]}'")
        except Exception as e:
            logger.error(f"❌ Erreur detect_intent: {e}")
            intent = "question"  # Par défaut
        
        # 4️⃣ Gérer les salutations et remerciements (réponses simples)
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
        
        # 5️⃣ Vérifier si on doit demander une clarification
        should_clarify, clarification = intelligent_chat.should_ask_clarification(
            req.message, detected_language
        )
        
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

        # 6️⃣ Récupérer le contexte du RAG (multiple documents) FILTRÉ par catégorie
        logger.info(f"🔍 Question: '{req.message}' | Langue: {detected_language} | Catégorie reçue: '{req.category}'")
        logger.info(f"   Type de category: {type(req.category)} | Repr: {repr(req.category)}")
        answer_raw, context = rag.ask(
            query=req.message, 
            k=5,
            language=detected_language,
            category=req.category,  # 🎯 FILTRAGE PAR CATÉGORIE
            min_confidence=0.40  # 🎯 Seuil de confiance (0.40 = équilibré)
        )
        
        # 7️⃣ Retourner directement la réponse du RAG (logique pure, avec LLaMA 2)
        return {
            "session_id": session_id,
            "conversation_id": None,
            "response": answer_raw,  # Réponse directe du RAG
            "language": detected_language,
            "intent": intent,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur du service AI: {str(e)}")


@router.post("/chat/intelligent")
def chat_intelligent(req: ChatRequest):
    """
    🧠 NOUVEAU: Endpoint avec IA VRAIMENT INTELLIGENTE
    
    Utilise Ollama (LLaMA 2 local) pour:
    - Analyser la question dans le contexte
    - Reformuler naturellement les réponses RAG
    - Maintenir un dialogue cohérent
    - Adapter au contexte burkinabè
    
    DIFFÉRENCE avec /chat/guest:
    - /chat/guest = RAG pur (copier-coller)
    - /chat/intelligent = RAG + LLM (dialogue intelligent)
    """
    try:
        # 1️⃣ NORMALISATION ET CORRECTION AUTOMATIQUE
        original_message = req.message
        normalized_message = text_normalizer.normalize(req.message)
        
        # Utiliser le message normalisé pour le traitement
        req.message = normalized_message
        
        # Log si correction effectuée
        if normalized_message != original_message:
            logger.info(f"✏️ Message corrigé: '{original_message}' → '{normalized_message}'")
        
        # 2️⃣ Session management
        session_id = req.session_id or str(uuid4())
        
        # 3️⃣ Langue: prioriser le choix utilisateur (pas d'auto-détection)
        detected_language = (req.language or "").strip() or "fr"
        
        try:
            intent = conversation_service.detect_intent(req.message, detected_language)
        except:
            intent = "question"
        
        # 4️⃣ DÉTECTER DÉCLARATIONS DE LANGUE (je parle français/moore/dioula)
        message_lower = req.message.lower()
        language_declaration_keywords = [
            "je parle français", "je parle francais", "je parque français",
            "je parle moore", "je parle mooré", "je parle moré",
            "je parle dioula", "je parle dyula",
            "en français", "en francais", "parle français"
        ]
        
        if any(keyword in message_lower for keyword in language_declaration_keywords):
            language_response = (
                "D'accord. Je te réponds en français.\n\n"
                "Pose-moi ta question (ex: plantes médicinales, karité/PFNL, savon, métiers, civisme, maths pratiques)."
            )
            
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", language_response)
            
            return {
                "session_id": session_id,
                "response": language_response,
                "language": "fr",
                "intent": "language_declaration",
                "mode": "language_preference",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 5️⃣ DÉTECTER DEMANDES D'EXEMPLES (montre-moi, donne exemple, je choisis)
        example_keywords = [
            "montre", "montre moi", "montre-moi", "exemple", "exemples",
            "donne exemple", "donne-moi exemple", "cite", "liste",
            "je choisis", "j'ai choisi", "je veux", "je voudrais",
            "parle moi de", "parle-moi de", "dis moi", "dis-moi"
        ]
        
        # Détecter le domaine mentionné
        domain_keywords = {
            "plantes": "Plantes Medicinales",
            "plante": "Plantes Medicinales",
            "médicinale": "Plantes Medicinales",
            "medicinale": "Plantes Medicinales",
            "remède": "Plantes Medicinales",
            "remede": "Plantes Medicinales",
            "santé": "Plantes Medicinales",
            "sante": "Plantes Medicinales",
            "maladie": "Plantes Medicinales",
            
            "agriculture": "Agriculture Locale",
            "cultiver": "Agriculture Locale",
            "culture": "Agriculture Locale",
            "mil": "Agriculture Locale",
            "sorgho": "Agriculture Locale",
            
            "savon": "Science Pratique - Saponification",
            "saponification": "Science Pratique - Saponification",
            
            "métier": "Metiers Informels",
            "metier": "Metiers Informels",
            "business": "Metiers Informels",
        }
        
        is_asking_example = any(keyword in message_lower for keyword in example_keywords)
        detected_domain = None
        
        for keyword, domain in domain_keywords.items():
            if keyword in message_lower:
                detected_domain = domain
                break
        
        # Si pas de domaine détecté mais demande d'exemple, utiliser la catégorie fournie
        if is_asking_example and not detected_domain:
            if req.category and req.category != "general":
                detected_domain = req.category
        
        # Si demande d'exemples + domaine détecté → Donner des exemples concrets
        if is_asking_example and detected_domain:
            # Exemples pré-définis par domaine
            domain_examples = {
                "Plantes Medicinales": (
                    "🌿 **Voici des plantes médicinales burkinabè que je connais:**\n\n"
                    "1. **Moringa** 🌱 - Combat la fatigue et l'anémie\n"
                    "   → Consommer 1 cuillère à soupe de poudre par jour\n\n"
                    "2. **Karité** 🥜 - Soins de la peau et cheveux\n"
                    "   → Beurre naturel pour hydrater et protéger\n\n"
                    "3. **Baobab** 🌳 - Riche en vitamine C\n"
                    "   → Poudre de fruit pour renforcer l'immunité\n\n"
                    "4. **Néré** 🌰 - Soumbala pour l'assaisonnement\n"
                    "   → Aide la digestion et riche en protéines\n\n"
                    "**Pose-moi une question précise sur une plante !**\n"
                    "Exemple: \"Comment utiliser le moringa contre la fatigue ?\""
                ),
                "Agriculture Locale": (
                    "🌾 **Voici des cultures importantes au Burkina Faso:**\n\n"
                    "1. **Mil** - Culture vivrière de base\n"
                    "   → Planter en début de saison des pluies\n\n"
                    "2. **Sorgho** - Résistant à la sécheresse\n"
                    "   → Bon pour le tô et le dolo\n\n"
                    "3. **Maïs** - Culture commerciale\n"
                    "   → Demande plus d'eau\n\n"
                    "4. **Niébé (haricot)** - Protéines végétales\n"
                    "   → Enrichit le sol en azote\n\n"
                    "**Pose une question précise !**\n"
                    "Exemple: \"Quelle est la meilleure période pour cultiver le mil ?\""
                ),
                "Science Pratique - Saponification": (
                    "🧴 **Je peux t'aider avec la fabrication de savon:**\n\n"
                    "- Savon à base de karité\n"
                    "- Savon noir traditionnel\n"
                    "- Saponification à froid\n"
                    "- Dosage de la soude caustique\n\n"
                    "**Pose une question !**\n"
                    "Exemple: \"Comment faire du savon au karité ?\""
                ),
                "Metiers Informels": (
                    "💼 **Voici des métiers informels au Burkina:**\n\n"
                    "- Transformation de produits locaux\n"
                    "- Petit commerce\n"
                    "- Artisanat\n"
                    "- Services à domicile\n\n"
                    "**Dis-moi ce qui t'intéresse !**"
                )
            }
            
            example_response = domain_examples.get(
                detected_domain,
                f"Je peux t'aider avec {detected_domain}. Pose-moi une question précise !"
            )
            
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", example_response)
            
            return {
                "session_id": session_id,
                "response": example_response,
                "language": detected_language,
                "intent": "request_examples",
                "mode": "examples_provided",
                "category": detected_domain,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 6️⃣ DÉTECTER QUESTIONS DE PRÉSENTATION (qui/nom/appelles)
        presentation_keywords = [
            "comment tu t'appel", "comment t'appel", "tu t'appel", 
            "c'est quoi ton nom", "quel est ton nom", "ton nom",
            "qui es tu", "qui es-tu", "tu es qui", "t'es qui",
            "comment tu", "qui tu es"
        ]
        
        if any(keyword in message_lower for keyword in presentation_keywords):
            presentation_response = (
                "Je m'appelle YINGR-AI ! 🇧🇫\n\n"
                "Je suis l'Intelligence Artificielle locale et souveraine du Burkina Faso. "
                "Mon rôle est de t'aider avec des connaissances pratiques sur:\n"
                "• Les plantes médicinales 🌿\n"
                "• L'agriculture locale 🌾\n"
                "• La transformation de produits 🧴\n"
                "• Les métiers informels 💼\n"
                "• Le civisme et le développement personnel 📚\n\n"
                "Comment puis-je t'aider aujourd'hui ?"
            )
            
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", presentation_response)
            
            return {
                "session_id": session_id,
                "response": presentation_response,
                "language": detected_language,
                "intent": "presentation",
                "mode": "intelligent_presentation",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 7️⃣ Gérer salutations (SANS LLM pour réponses plus rapides et consistantes)
        if intent == "greeting":
            greeting_responses_fr = [
                "Bonjour ! Je suis YINGR-AI, ton assistant burkinabè. 🇧🇫\n\nComment puis-je t'aider aujourd'hui ?",
                "Salut ! Content de te parler. 😊\n\nQue veux-tu savoir ?",
                "Bienvenue ! Je suis là pour t'aider. 👋\n\nPose-moi tes questions sur l'agriculture, la santé, les métiers..."
            ]
            
            import random
            greeting_response = random.choice(greeting_responses_fr)
            
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", greeting_response)
            
            return {
                "session_id": session_id,
                "response": greeting_response,
                "language": detected_language,
                "intent": intent,
                "mode": "intelligent_greeting",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 8️⃣ Gérer remerciements (SANS LLM pour réponses plus rapides)
        if intent == "thanks":
            thanks_responses_fr = [
                "Je t'en prie ! 😊 N'hésite pas si tu as d'autres questions.",
                "Avec plaisir ! Je suis là pour t'aider. 🙌",
                "Pas de souci ! Reviens quand tu veux. 👍"
            ]
            
            import random
            thanks_response = random.choice(thanks_responses_fr)
            
            ai_brain.add_to_history("user", req.message)
            ai_brain.add_to_history("assistant", thanks_response)
            
            return {
                "session_id": session_id,
                "response": thanks_response,
                "language": detected_language,
                "intent": intent,
                "mode": "intelligent_thanks",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 9️⃣ COMPRENDRE L'INTENTION de la question
        logger.info(f"🧠 Question originale: '{req.message}'")
        
        # Essayer de comprendre la question (surtout pour santé)
        understanding = QueryUnderstanding.understand_health_query(req.message)
        if understanding:
            logger.info(f"💡 Compréhension: {understanding['suggestion']}")
            # Utiliser la requête reformulée
            expanded_query = understanding['reformulated_query']
        else:
            # Enrichir la question avec des mots-clés et synonymes
            expanded_query = req.message
            
            # Ajouter des mots-clés selon le contexte
            query_lower = req.message.lower()
            
            # Problèmes digestifs (estomac, gaz, ballonnement, digestion...)
            if any(word in query_lower for word in ['maux', 'mal', 'douleur', 'soigner', 'traiter', 'estomac', 'ventre', 'gaz', 'ballonnement', 'digestion', 'intestin', 'gastrique']):
                # Ajouter des termes médicaux locaux + synonymes
                expanded_query += " plantes médicinales traditionnelles Burkina traitement naturel remède estomac ventre digestion gastrique"
            
            # Fabrication savon
            elif any(word in query_lower for word in ['savon', 'fabriquer', 'saponification', 'lessive']):
                expanded_query += " fabrication artisanale transformation saponification recette savon"
            
            # Karité et PFNL
            elif any(word in query_lower for word in ['karité', 'beurre', 'noix', 'pfnl']):
                expanded_query += " transformation PFNL beurre karité production artisanale"
            
            # Maladies et symptômes généraux
            elif any(word in query_lower for word in ['fièvre', 'toux', 'rhume', 'paludisme', 'malade']):
                expanded_query += " plantes médicinales santé traitement naturel Burkina remède"
        
        logger.info(f"🔍 Question enrichie: '{expanded_query}'")
        
        # Interroger le RAG avec la requête enrichie
        answer_raw, context_raw = rag.ask(
            query=expanded_query,  # ← Utiliser la requête ENRICHIE
            k=10,
            language=detected_language,
            category=req.category,
            min_confidence=0.15
        )
        
        logger.info(f"📊 RAG résultats: answer_raw length={len(answer_raw) if answer_raw else 0}, context_raw length={len(context_raw) if context_raw else 0}")
        if isinstance(context_raw, str):
            logger.info(f"📄 Contexte brut (100 premiers chars): {context_raw[:100] if context_raw else 'VIDE'}")
        else:
            logger.info(f"📄 Contexte brut (type={type(context_raw)}): {str(context_raw)[:100] if context_raw else 'VIDE'}")
        
        # 9️⃣ Transformer le contexte en format adapté pour AI Brain
        # RAGService.ask() renvoie un contexte texte avec séparateur "\n\n---\n\n".
        # Ce contexte est souvent "answers-only" (sans questions), donc on construit
        # des pseudo-sources Q/R en réutilisant la question utilisateur.
        rag_results = []
        for block in _rag_context_to_blocks(context_raw)[:3]:
            rag_results.append({
                "question": req.message,
                "reponse": block
            })
        
        logger.info(f"📚 {len(rag_results)} documents structurés pour le LLM")
        
        # 🔟 🎯 GÉNÉRATION INTELLIGENTE avec AI Brain
        intelligent_response = ai_brain.generate_intelligent_response(
            question=req.message,
            rag_results=rag_results,
            category=req.category,
            language=detected_language
        )
        
        # 1️⃣1️⃣ 🔊 GÉNÉRATION AUDIO (uniquement pour mooré et dioula)
        audio_url = None
        audio_mode = "not_available"
        
        if detected_language in ["mo", "di"]:  # Mooré ou Dioula
            try:
                response_text = intelligent_response["reponse"]
                audio_url, audio_mode = tts_service.generate_audio(
                    text=response_text,
                    language=detected_language
                )
                logger.info(f"🔊 Audio généré: {audio_url} (mode: {audio_mode})")
            except Exception as e:
                logger.warning(f"⚠️ Audio non disponible: {e}")
                audio_url = None
                audio_mode = "not_available"
        
        # 1️⃣2️⃣ Retourner la réponse intelligente avec audio
        response_text = _fix_mojibake(intelligent_response["reponse"])
        context_first = _fix_mojibake(rag_results[0]["reponse"]) if rag_results else ""

        payload = {
            "session_id": session_id,
            "response": response_text,
            "language": detected_language,
            "intent": intent,
            "category": intelligent_response["categorie"],
            "sources_count": intelligent_response.get("sources_utilisees", 0),
            "mode": intelligent_response.get("mode", "intelligent"),
            "context": [context_first] if context_first else [],  # Première source
            "timestamp": intelligent_response.get("timestamp", datetime.utcnow().isoformat()),
            "audio_url": audio_url,
            "audio_mode": audio_mode
        }

        # Certains clients (ex: PowerShell Invoke-WebRequest) affichent des accents cassés
        # si le charset n'est pas précisé. On force UTF-8 pour un rendu correct.
        return JSONResponse(content=payload, media_type="application/json; charset=utf-8")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Erreur chat intelligent: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur service AI intelligent: {str(e)}")


@router.post("/chat/clear-history")
def clear_chat_history():
    """Efface l'historique conversationnel (pour tests)"""
    try:
        ai_brain.clear_history()
        return {"status": "ok", "message": "Historique effacé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None,
    category: Optional[str] = "general",
    language: Optional[str] = "fr"  # Langue choisie par l'utilisateur
):
    """
    Endpoint pour envoyer un message VOCAL (Speech-to-Text puis dialogue intelligent)
    
    Flux complet: Audio → STT (Whisper) → Texte → RAG+LLM → Texte → TTS → Audio
    
    Permet aux utilisateurs de parler en mooré/dioula sans taper les caractères spéciaux (ɔ, ɛ, etc.)
    """
    try:
        logger.info("=" * 60)
        logger.info("🎤 NOUVELLE REQUÊTE VOCALE")
        logger.info("=" * 60)
        
        # 1️⃣ Vérifier que STT est disponible
        if not stt_service.is_available():
            logger.error("❌ Service STT non disponible")
            raise HTTPException(
                status_code=503,
                detail="Service de reconnaissance vocale non disponible. Installer Whisper: pip install openai-whisper"
            )
        
        logger.info("✅ Service STT disponible")
        
        # 2️⃣ Lire les données audio
        logger.info(f"📥 Réception audio: {audio.filename} ({audio.content_type})")
        audio_bytes = await audio.read()
        logger.info(f"📊 Taille audio: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.1f} KB)")
        
        if len(audio_bytes) == 0:
            logger.error("❌ Fichier audio vide")
            raise HTTPException(status_code=400, detail="Fichier audio vide")
        
        if len(audio_bytes) < 1000:  # Moins de 1KB = probablement invalide
            logger.warning(f"⚠️ Audio très court: {len(audio_bytes)} bytes")
        
        # 3️⃣ Transcription Speech-to-Text avec Whisper
        logger.info(f"🔄 Lancement transcription Whisper (langue: {language})...")
        
        try:
            # Utiliser la langue choisie par l'utilisateur au lieu de l'auto-détection
            transcription, detected_language, confidence = stt_service.transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                filename=audio.filename,
                language=language  # Utiliser la langue choisie
            )
        except Exception as e:
            logger.error(f"❌ Erreur transcription Whisper: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la transcription: {str(e)}"
            )
        
        logger.info(f"📝 Transcription brute: '{transcription}' (longueur: {len(transcription)})")

        # Si Whisper n'a rien compris, renvoyer une réponse gentille plutôt qu'une erreur 400
        if not transcription or len(transcription.strip()) == 0:
            logger.error("❌ Transcription vide, Whisper n'a rien compris")
            return {
                "session_id": session_id or f"voice_{uuid4().hex[:8]}",
                "transcription": "",
                "transcription_confidence": 0.0,
                "response": (
                    "Je n'ai pas bien entendu ce que tu as dit. "
                    "Peux-tu répéter en parlant un peu plus fort et pendant 3 à 5 secondes ?"
                ),
                "language": language or "fr",
                "intent": "incomprehensible_audio",
                "category": category or "general",
                "sources_count": 0,
                "mode": "voice_intelligent",
                "context": [],
                "audio_url": None,
                "audio_mode": "not_available",
            }

        logger.info(f"✅ Transcription réussie: '{transcription}' (langue: {detected_language}, confiance: {confidence:.2%})")
        
        # 4️⃣ Traiter le texte transcrit avec l'endpoint intelligent
        # Utiliser le même flux que /chat/intelligent
        
        # Créer une requête interne
        from pydantic import BaseModel
        
        class InternalChatRequest(BaseModel):
            message: str
            session_id: Optional[str] = None
            category: Optional[str] = "general"
        
        internal_req = InternalChatRequest(
            message=transcription,
            session_id=session_id,
            category=category,
            language=language  # Utiliser la langue choisie
        )
        
        # 5️⃣ Appeler la logique du chat intelligent
        # (On réutilise le même code que /chat/intelligent)
        
        # Générer session_id si nécessaire
        if not session_id:
            session_id = f"voice_{uuid4().hex[:8]}"
        
        # Normaliser le texte (correction typos)
        normalized_message = text_normalizer.normalize(transcription)
        logger.info(f"📝 Message normalisé: '{normalized_message}'")

        # Utiliser la langue choisie par l'utilisateur (pas d'auto-détection)
        detected_lang = language
        intent = conversation_service.detect_intent(normalized_message, detected_lang)

        # 📌 Cas spécial : salutations vocales
        if intent == "greeting":
            logger.info("🙋 Intent vocal détecté: greeting – réponse d'accueil sans RAG")

            greeting_text = conversation_service.generate_greeting_response(detected_lang)

            audio_url = None
            audio_mode = "not_available"
            if detected_lang in ["mo", "di"]:
                try:
                    audio_url, audio_mode = tts_service.generate_audio(
                        text=greeting_text,
                        language=detected_lang
                    )
                    logger.info(f"🔊 Audio réponse greeting généré: {audio_url} (mode: {audio_mode})")
                except Exception as e:
                    logger.warning(f"⚠️ Audio réponse greeting non disponible: {e}")

            return {
                "session_id": session_id,
                "transcription": transcription,
                "transcription_confidence": confidence,
                "response": greeting_text,
                "language": detected_lang,
                "intent": intent,
                "category": category,
                "sources_count": 0,
                "mode": "voice_greeting",
                "context": [],
                "timestamp": datetime.utcnow().isoformat(),
                "audio_url": audio_url,
                "audio_mode": audio_mode,
                "stt_service": "whisper",
                "workflow": "voice → stt → greeting"
            }

        # Interroger RAG pour les autres intents
        answer_raw, context_raw = rag.ask(
            query=normalized_message,
            k=3,
            language=detected_lang,
            category=category,
            min_confidence=0.35
        )

        # Transformer contexte RAG
        rag_results = []
        for block in _rag_context_to_blocks(context_raw)[:3]:
            rag_results.append({
                "question": normalized_message,
                "reponse": block
            })

        # Génération intelligente avec AI Brain
        intelligent_response = ai_brain.generate_intelligent_response(
            question=normalized_message,
            rag_results=rag_results,
            category=category,
            language=detected_lang
        )
        
        # 6️⃣ Génération audio de la réponse (TTS)
        audio_url = None
        audio_mode = "not_available"
        
        if detected_lang in ["mo", "di"]:
            try:
                response_text = intelligent_response["reponse"]
                audio_url, audio_mode = tts_service.generate_audio(
                    text=response_text,
                    language=detected_lang
                )
                logger.info(f"🔊 Audio réponse généré: {audio_url} (mode: {audio_mode})")
            except Exception as e:
                logger.warning(f"⚠️ Audio réponse non disponible: {e}")
        
        # 7️⃣ Retourner la réponse complète
        return {
            "session_id": session_id,
            "transcription": transcription,  # ← Texte transcrit
            "transcription_confidence": confidence,
            "response": intelligent_response["reponse"],
            "language": detected_lang,
            "intent": intent,
            "category": intelligent_response["categorie"],
            "sources_count": intelligent_response.get("sources_utilisees", 0),
            "mode": "voice_intelligent",  # Mode spécial pour voix
            "context": [rag_results[0]["reponse"]] if rag_results else [],
            "timestamp": intelligent_response.get("timestamp", datetime.utcnow().isoformat()),
            "audio_url": audio_url,  # ← Audio de la réponse
            "audio_mode": audio_mode,
            "stt_service": "whisper",
            "workflow": "voice → stt → rag+llm → tts → voice"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ Erreur chat vocal: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur service chat vocal: {str(e)}")

