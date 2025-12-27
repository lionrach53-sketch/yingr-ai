"""
AI BRAIN – Cerveau conversationnel YINGR-AI 🇧🇫

PRODUCTION READY:
- Historique conversation persistant dans Redis
- Cache RAG ultra rapide avec monitoring
- Limitation longueur RAG + max 3 sources
- Multi-langues: fr / mo / di
- CPU only, fallback robuste
"""

import os
import requests
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class AIBrain:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = os.getenv("OLLAMA_URL", ollama_url)
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        
        # Conversation history locale
        self.conversation_history: List[Dict] = []
        self.max_history = 5

        # Cache Redis
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        self.redis_client: Optional[redis.Redis] = None
        self.cache_hits = 0
        self.cache_misses = 0
        self._init_redis_cache()

    # ------------------- REDIS CACHE -------------------
    def _init_redis_cache(self):
        self.redis_client = None
        if not REDIS_AVAILABLE:
            print("⚠️  Redis non installé - cache désactivé")
            return
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            print("✅ Cache Redis actif")
        except Exception as e:
            print(f"⚠️  Redis indisponible ({e}) - cache désactivé")
            self.redis_client = None

    def _make_cache_key(self, question: str, language: str) -> str:
        key_data = f"{question.lower().strip()}:{language}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"yingr_rag:{key_hash}"

    def _get_cached_response(self, question: str, language: str) -> Optional[Dict]:
        if not self.redis_client:
            return None
        try:
            cache_key = self._make_cache_key(question, language)
            cached = self.redis_client.get(cache_key)
            if cached:
                self.cache_hits += 1
                # Charger historique si présent
                data = json.loads(cached)
                hist = data.get("history", [])
                if hist:
                    self.conversation_history = hist[-self.max_history*2:]
                return data
        except Exception as e:
            print(f"⚠️  Erreur lecture cache: {e}")
        return None

    def _set_cached_response(self, question: str, language: str, response: Dict) -> None:
        if not self.redis_client:
            return
        try:
            cache_key = self._make_cache_key(question, language)
            # Sauvegarder aussi l'historique
            response_copy = response.copy()
            response_copy["history"] = self.conversation_history
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(response_copy))
        except Exception as e:
            print(f"⚠️  Erreur écriture cache: {e}")

    # ------------------- HISTORIQUE -------------------
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def clear_history(self):
        self.conversation_history = []

    def get_context_summary(self) -> str:
        if not self.conversation_history:
            return ""
        summary = "\n=== HISTORIQUE ===\n"
        for msg in self.conversation_history[-6:]:
            role_fr = "Utilisateur" if msg["role"] == "user" else "Assistant"
            summary += f"{role_fr}: {msg['content']}\n"
        summary += "=== FIN ===\n\n"
        return summary

    # ------------------- STATISTIQUES CACHE -------------------
    def get_cache_stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "redis_active": self.redis_client is not None
        }

    # ------------------- GENERATION REPONSE -------------------
    def generate_response(self, question: str, rag_results: List[Dict], language: str, category: str = "general") -> Dict:
        cached_response = self._get_cached_response(question, language)
        if cached_response:
            self.add_to_history("user", question)
            self.add_to_history("assistant", cached_response.get("reponse", ""))
            return {**cached_response, "cache_hit": True}

        if not rag_results:
            return self._no_data_response(question, language, category)

        system_prompt, user_prompt = self._build_prompts(question, rag_results, language)

        try:
            self.cache_misses += 1
            response = self._call_ollama(system_prompt, user_prompt)
            result = {
                "reponse": response,
                "mode": "intelligent",
                "langue": language,
                "categorie": category,
                "sources_utilisees": len(rag_results),
                "sources": len(rag_results),
                "timestamp": datetime.utcnow().isoformat(),
                "cache_hit": False
            }
            self.add_to_history("user", question)
            self.add_to_history("assistant", response)
            self._set_cached_response(question, language, result)
            return result
        except Exception as e:
            fallback = self._fallback_response(question, rag_results, language)
            return {
                "reponse": fallback,
                "mode": "structured_rag",
                "erreur": str(e),
                "langue": language,
                "categorie": category,
                "sources_utilisees": len(rag_results),
                "sources": len(rag_results),
                "timestamp": datetime.utcnow().isoformat(),
                "cache_hit": False
            }

    def generate_intelligent_response(self, question: str, rag_results: List[Dict], category: str = "general", language: str = "fr") -> Dict:
        return self.generate_response(question, rag_results, language, category)

    # ------------------- PROMPTS -------------------
    def _build_prompts(self, question: str, rag_results: List[Dict], language: str) -> Tuple[str, str]:
        """Construit les prompts système/utilisateur à partir de briques pédagogiques.

        Compatible avec deux formats de RAG:
        - Ancien: {"question": str, "reponse": str}
        - Enrichi: {"reponse_courte", "reponse_detaillee", "conseil", "avertissement", ...}
        """

        blocks = []
        for r in rag_results[:3]:
            # Nouveau format enrichi UNIQUEMENT
            short = (r.get("reponse_courte") or "").strip()
            detailed = (r.get("reponse_detaillee") or "").strip()
            advice = (r.get("conseil") or "").strip()
            warning = (r.get("avertissement") or "").strip()

            if not (short or detailed or advice or warning):
                # Ancien champ "reponse" n'est plus utilisé
                continue

            # Construire un petit bloc structuré pour le LLM
            lines = []
            if short:
                lines.append(f"Idée principale : {short}")
            if detailed:
                # Limiter la taille pour éviter de saturer le contexte
                detail_trimmed = detailed[:500] + "..." if len(detailed) > 500 else detailed
                lines.append(f"Explication : {detail_trimmed}")
            if advice:
                lines.append(f"Conseil pratique : {advice}")
            if warning:
                lines.append(f"Avertissement : {warning}")

            blocks.append("\n".join(lines))

        knowledge = "\n\n---\n\n".join(blocks)
        history_context = self.get_context_summary()

        system_prompts = {
            "fr": (
                "Tu es YINGR-AI, assistant burkinabè.\n"
                "Commence par reformuler en une phrase la situation de la personne (lieu, niveau d'étude, activité si mentionnée).\n"
                "Explique simplement pour un public burkinabè, en tenant compte du contexte de vie au Burkina (ville, village, niveau de revenu).\n"
                "Compose ta réponse à partir des briques fournies (idée principale, explication, conseil, avertissement) en évitant de répéter plusieurs fois la même idée.\n"
                "Utilise uniquement les informations qui répondent directement à la question posée ; ignore les briques qui parlent d'autres thèmes (par exemple citoyenneté, politique, sujets trop éloignés).\n"
                "Si la question contient plusieurs questions différentes, réponds à chacune séparément de manière structurée (1., 2., 3.), avant ta question finale.\n"
                "Propose ensuite 2 à 4 pistes concrètes ou options adaptées à la situation décrite, sous forme de phrases courtes et pratiques.\n"
                "Ne récite pas les textes tels quels, reformule de manière naturelle.\n"
                "Réponds globalement en 3 à 7 phrases maximum (hors ta question finale).\n"
                "Termine toujours par UNE seule question courte à l'utilisateur pour continuer le dialogue.\n"
                "Langue: français simple et naturel."
            ),
            "mo": "Fo yaa YINGR-AI.\nFo tɩ yel n be t'a sũur sũur.\nFo tɩ kãnga woto.\nGom: mooré.",
            "di": "I ye YINGR-AI ye.\nI ka kuma sùrun ani ɲɔgɔn.\nI ka jaabi kelen di.\nKan: dioula."
        }

        system_prompt = system_prompts.get(language, system_prompts["fr"])

        user_prompt = f"""{history_context}Voici des éléments de connaissance issus de la base :

    {knowledge}

    Ta mission :
    - Synthétiser l'idée principale.
    - Donner une explication simple et concrète adaptée au Burkina Faso.
    - Ajouter au moins un conseil pratique si disponible.
    - Mentionner un avertissement si pertinent.
    - Utiliser uniquement les éléments qui aident vraiment à répondre à la question (ignorer les informations hors-sujet par rapport à la demande de l'utilisateur).
    - Ne répète pas la question mot pour mot.
    - Termine par UNE seule question pour continuer le dialogue.

    Question utilisateur : {question}
    """
        return system_prompt, user_prompt

    # ------------------- OLLAMA -------------------
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature":0.4, "top_p":0.85, "num_predict":100, "num_ctx":1024, "repeat_penalty":1.2}
        }
        r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}).get("content") or "").strip()

    # ------------------- FALLBACK -------------------
    def _fallback_response(self, question: str, rag_results: List[Dict], language: str) -> str:
        bullets = []
        for r in rag_results[:3]:
            short = (r.get("reponse_courte") or "").strip()
            detailed = (r.get("reponse_detaillee") or "").strip()
            # Préférer la courte, sinon un extrait de la détaillée
            if short:
                bullets.append(f"- {short}")
            elif detailed:
                detail_trimmed = detailed[:200] + "..." if len(detailed) > 200 else detailed
                bullets.append(f"- {detail_trimmed}")
        if not bullets:
            bullets = ["- Je n'ai pas assez d'informations structurées pour répondre précisément."]
        base = "\n".join(bullets)
        questions = {"fr":"Laquelle de ces pistes vous intéresse le plus ?","mo":"Yembã bãnga ne fo kẽe ?","di":"Min bɛ i fɛ kosɛbɛ ?"}
        return f"{base}\n\n{questions.get(language, questions['fr'])}"

    # ------------------- NO DATA -------------------
    def _no_data_response(self, question: str, language: str, category: str = "general") -> Dict:
        messages = {
            "fr":"Je n'ai pas assez d'informations dans ma base.\nPouvez-vous préciser votre question ou donner un exemple ?",
            "mo":"Tɩɩs sã n tɩ laar ra.\nFo tõe yel makre sã fo kẽe ?",
            "di":"Kunnafoni tɛ se ka ɲɔgɔn.\nI bɛ se ka kuma wɛrɛ fɔ wa ?"
        }
        return {
            "reponse": messages.get(language, messages["fr"]),
            "mode": "no_rag",
            "langue": language,
            "categorie": category,
            "sources_utilisees":0,
            "timestamp": datetime.utcnow().isoformat()
        }


# INSTANCE GLOBALE
ai_brain = AIBrain()
