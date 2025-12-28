"""
AI BRAIN – Cerveau conversationnel YINGR-AI 🇧🇫

AMÉLIORATIONS :
- Réponses conversationnelles courtes en premier
- Suggestions de dialogue pour poursuivre
- Réponses adaptées au contexte (salutations, remerciements)
- Réponses structurées mais optimisées pour le chat
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

    # ------------------- GÉNÉRATION RÉPONSE AMÉLIORÉE -------------------
    def generate_response(self, question: str, rag_results: List[Dict], language: str, category: str = "general") -> Dict:
        """Version améliorée pour des réponses plus conversationnelles"""
        
        # Détection d'intention simple
        question_lower = question.lower().strip()
        if self._is_greeting(question_lower, language):
            return self._generate_greeting_response(language, category)
        
        if self._is_thanks(question_lower, language):
            return self._generate_thanks_response(language, category)
        
        if self._is_simple_question(question_lower):
            return self._generate_concise_response(question, rag_results, language, category)
        
        # Pour les questions complexes, utiliser le cache
        cached_response = self._get_cached_response(question, language)
        if cached_response:
            self.add_to_history("user", question)
            self.add_to_history("assistant", cached_response.get("reponse", ""))
            return {**cached_response, "cache_hit": True}

        # Génération de réponse intelligente avec structure optimisée
        self.cache_misses += 1
        system_prompt, user_prompt = self._build_conversational_prompts(question, rag_results or [], language)
        
        try:
            response = self._call_ollama(system_prompt, user_prompt)
            
            if response and len(response.strip()) > 30:
                # Extraire la réponse principale et les suggestions
                main_response, suggestions = self._extract_main_response_and_suggestions(response, language)
                
                result = {
                    "reponse": main_response,
                    "suggestions": suggestions[:3],  # Limiter à 3 suggestions max
                    "mode": "conversational",
                    "langue": language,
                    "categorie": category,
                    "sources_utilisees": len(rag_results or []),
                    "sources": len(rag_results or []),
                    "timestamp": datetime.utcnow().isoformat(),
                    "cache_hit": False,
                    "full_response": response  # Garder la réponse complète pour référence
                }
                
                self.add_to_history("user", question)
                self.add_to_history("assistant", main_response)
                self._set_cached_response(question, language, result)
                return result
            
            # Fallback si pas de réponse générée
            return self._fallback_conversational_response(question, rag_results or [], language, category)
            
        except Exception as e:
            return self._fallback_conversational_response(question, rag_results or [], language, category)

    def generate_intelligent_response(self, question: str, rag_results: List[Dict], category: str = "general", language: str = "fr") -> Dict:
        return self.generate_response(question, rag_results, language, category)

    # ------------------- DÉTECTION D'INTENTIONS -------------------
    def _is_greeting(self, question: str, language: str) -> bool:
        greetings = {
            "fr": ["bonjour", "salut", "coucou", "hello", "bjr", "slt", "bonsoir"],
            "mo": ["kɩbare", "ne y taabo", "ne y taare", "ne y windga"],
            "di": ["i ni sɔgɔma", "i ni tile", "i ni wula", "i ni su"]
        }
        lang_greetings = greetings.get(language, greetings["fr"])
        return any(greet in question for greet in lang_greetings)

    def _is_thanks(self, question: str, language: str) -> bool:
        thanks = {
            "fr": ["merci", "remerci", "merci beaucoup", "merci bien"],
            "mo": ["barka", "barika", "a bɩɩn", "a yiib"],
            "di": ["a ni ce", "i ni ce", "a barika", "i ni ce ka"]
        }
        lang_thanks = thanks.get(language, thanks["fr"])
        return any(th in question for th in lang_thanks)

    def _is_simple_question(self, question: str) -> bool:
        simple_patterns = ["ok", "d'accord", "compris", "entendu", "super", "génial", "parfait"]
        return any(pattern in question for pattern in simple_patterns)

    # ------------------- RÉPONSES COURTES -------------------
    def _generate_greeting_response(self, language: str, category: str) -> Dict:
        greetings = {
            "fr": "Bonjour ! Je suis votre assistant YINGR-AI. Comment puis-je vous aider aujourd'hui ?",
            "mo": "Kɩbare ! Mam yaa YINGR-AI. Tõnd nonglem maana yaa ?",
            "di": "I ni sɔgɔma ! N ye YINGR-AI ye. N bɛ se ka i dɛmɛ di cogo jumɛn na ?"
        }
        return {
            "reponse": greetings.get(language, greetings["fr"]),
            "mode": "greeting",
            "langue": language,
            "categorie": category,
            "sources_utilisees": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _generate_thanks_response(self, language: str, category: str) -> Dict:
        thanks = {
            "fr": "Je vous en prie ! N'hésitez pas si vous avez d'autres questions.",
            "mo": "Barka ! Kãadem b sã yɩɩ n kɩt yõodo.",
            "di": "Baaraka ! Aw bɛna ɲininkali wɛrɛw kɛ wa, i k'a fɔ."
        }
        return {
            "reponse": thanks.get(language, thanks["fr"]),
            "mode": "thanks",
            "langue": language,
            "categorie": category,
            "sources_utilisees": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _generate_concise_response(self, question: str, rag_results: List[Dict], language: str, category: str) -> Dict:
        """Réponse concise pour des questions simples"""
        if rag_results and len(rag_results) > 0:
            # Prendre la première réponse courte si disponible
            short_responses = [r.get("reponse_courte", "").strip() for r in rag_results[:1] if r.get("reponse_courte")]
            if short_responses:
                response = short_responses[0]
            else:
                response = "👍 Parfait ! Voulez-vous approfondir ce sujet ou passer à autre chose ?"
        else:
            confirmations = {
                "fr": "👍 Parfait ! Voulez-vous approfondir ce sujet ou passer à autre chose ?",
                "mo": "👍 Yɩɩ sõma ! Fo sẽn tõog n bas tɩ yel woto wa tɩ tʋm taaba ?",
                "di": "👍 A ka ɲi ! Yala i b'a fɛ ka kuma in jigin wa, walima kuma wɛrɛw la ?"
            }
            response = confirmations.get(language, confirmations["fr"])
        
        return {
            "reponse": response,
            "suggestions": self._generate_suggestions(rag_results, language),
            "mode": "concise",
            "langue": language,
            "categorie": category,
            "sources_utilisees": len(rag_results or []),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ------------------- PROMPTS CONVERSATIONNELS -------------------
    def _build_conversational_prompts(self, question: str, rag_results: List[Dict], language: str) -> Tuple[str, str]:
        """Construit des prompts pour des réponses conversationnelles"""
        
        # Extraire les informations clés du RAG
        knowledge_blocks = []
        for r in rag_results[:3]:
            short = (r.get("reponse_courte") or "").strip()
            detailed = (r.get("reponse_detaillee") or "").strip()
            
            if short:
                knowledge_blocks.append(f"• {short}")
            elif detailed:
                # Extraire les 2 premières phrases
                sentences = detailed.split('.')
                if len(sentences) > 2:
                    summary = '.'.join(sentences[:2]) + '.'
                else:
                    summary = detailed[:150] + "..." if len(detailed) > 150 else detailed
                knowledge_blocks.append(f"• {summary}")
        
        knowledge = "\n".join(knowledge_blocks[:5])  # Limiter à 5 points max
        history_context = self.get_context_summary()

        system_prompts = {
            "fr": (
                "Tu es YINGR-AI, un assistant conversationnel pour le Burkina Faso.\n\n"
                "RÈGLES DE CONVERSATION :\n"
                "1. Réponds de façon NATURELLE et CONVERSATIONNELLE\n"
                "2. Commence par une réponse COURTE (1-2 phrases maximum)\n"
                "3. Propose ensuite 2-3 questions de suivi utiles\n"
                "4. Utilise le contexte mais ne le répète pas mot à mot\n"
                "5. Sois utile, précis et encourageant\n"
                "6. Adapte ton langage au public burkinabè\n\n"
                "FORMAT DE RÉPONSE :\n"
                "[Réponse courte et naturelle]\n\n"
                "📌 Pour continuer :\n"
                "1. [Première suggestion]\n"
                "2. [Deuxième suggestion]\n"
                "3. [Troisième suggestion]"
            ),
            "mo": (
                "Fo yaa YINGR-AI, bool nonglem soaba.\n\n"
                "GOMSE :\n"
                "1. Kãn-wẽng bɩ a ka soab a taab ye\n"
                "2. Jaabi tɩ yaa tãagre (yembã fãa a yiib)\n"
                "3. Pʋɩɩse sãmb 2 wa 3 tɩ yaa sugr ne f meng ye\n"
                "4. Tʋm tõnd tagmasgã la a ra tɩ wa tʋg n pʋgẽ ye\n"
                "5. Yaa boolma, yɩɩme n ta yãnde\n"
                "6. Gom n bas Burkina Faso soabã ye"
            ),
            "di": (
                "I ye YINGR-AI ye, dɛmɛbaga ye Burkina Faso.\n\n"
                "KAN SIRI :\n"
                "1. Jaabi ka ɲɛnamaya ani kumakan\n"
                "2. Jaabi dɔɔnin-dɔɔnin (ɲɔgɔn fɔlɔ kelen wa fila)\n"
                "3. ɲininkali 2 wa 3 di minnu bɛ se ka i ɲɛsin\n"
                "4. K'o t'a jira o jira, k'a sɔr fɛn wɛrɛw fɛ\n"
                "5. Ka dɛmɛ di, ka dɔn ko ɲɛ, ka dɛsɛ\n"
                "6. Kan fɔ Burkinabèw ye"
            )
        }

        system_prompt = system_prompts.get(language, system_prompts["fr"])

        user_prompt = f"""{history_context}

INFORMATIONS DISPONIBLES :
{knowledge if knowledge else "Aucune information spécifique trouvée."}

QUESTION DE L'UTILISATEUR :
{question}

GÉNÈRE une réponse conversationnelle courte suivie de suggestions pour continuer le dialogue."""

        return system_prompt, user_prompt

    # ------------------- EXTRACTION RÉPONSE ET SUGGESTIONS -------------------
    def _extract_main_response_and_suggestions(self, response: str, language: str) -> Tuple[str, List[str]]:
        """Extrait la réponse principale et les suggestions du texte généré"""
        
        lines = response.split('\n')
        main_response_lines = []
        suggestions = []
        
        in_suggestions = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Détecter le début des suggestions
            suggestion_markers = {
                "fr": ["📌", "➡️", "💡", "pour continuer", "suggestion", "question"],
                "mo": ["📌", "➡️", "💡", "tɩ pʋɩɩse", "sugr", "sãmblem"],
                "di": ["📌", "➡️", "💡", "ka taa ɲɛ", "siri", "ɲininkali"]
            }
            
            markers = suggestion_markers.get(language, suggestion_markers["fr"])
            if any(marker in line.lower() for marker in markers):
                in_suggestions = True
            
            # Si on est dans les suggestions
            if in_suggestions:
                # Détecter les éléments de liste
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '- ', '• ', '▶ ', '✓ ')):
                    # Nettoyer la suggestion
                    clean_suggestion = line[2:].strip() if line[0].isdigit() else line[2:].strip() if line.startswith('- ') else line[1:].strip()
                    if clean_suggestion:
                        suggestions.append(clean_suggestion)
                elif line and not line.startswith('==='):  # Éviter les séparateurs
                    suggestions.append(line)
            else:
                # Ajouter à la réponse principale (sauf les marqueurs)
                if not any(marker in line.lower() for marker in markers):
                    main_response_lines.append(line)
        
        # Si pas de suggestions détectées, en générer automatiquement
        if not suggestions:
            suggestions = self._generate_suggestions([], language)
        
        # Nettoyer la réponse principale
        main_response = ' '.join(main_response_lines).strip()
        if len(main_response) > 500:  # Limiter la taille
            main_response = main_response[:497] + "..."
        
        return main_response, suggestions[:3]  # Limiter à 3 suggestions

    def _generate_suggestions(self, rag_results: List[Dict], language: str) -> List[str]:
        """Génère des suggestions de suivi basées sur le contexte"""
        
        suggestions_dict = {
            "fr": [
                "Pouvez-vous me donner plus de détails ?",
                "Quels sont les meilleurs conseils pratiques ?",
                "Y a-t-il des pièges à éviter ?",
                "Comment appliquer cela dans ma situation ?",
                "Quelles sont les alternatives possibles ?"
            ],
            "mo": [
                "Fo tõog n maan f meng fãa n yɩɩle ?",
                "Tõnd pʋgẽ yaa bõe-y ye ?",
                "Bɩ yaa bõe ne fo sagl n tɩ pa wẽng n tʋm ?",
                "Tɩ maan bʋɩl-woto fo leb n be pʋgẽ ?",
                "Yembã bãnga ne tɩ tʋm yel-woto ?"
            ],
            "di": [
                "Yala i bɛ se ka fɛn caman fɔ wa ?",
                "Dɛmɛ minnu ka di kosɛbɛ ?",
                "Yala fɛn minnu ka kan ka kɛ wa ?",
                "N bɛ se ka o kɛ cogo jumɛn na n yɛrɛ la ?",
                "Yala fɛn wɛrɛw bɛ yen wa ?"
            ]
        }
        
        # Suggestions spécifiques basées sur le RAG
        specific_suggestions = []
        for r in rag_results[:2]:
            short = r.get("reponse_courte", "").strip()
            if short:
                # Créer une suggestion basée sur le contenu
                if language == "fr":
                    specific_suggestions.append(f"En savoir plus sur : {short[:50]}...")
                elif language == "mo":
                    specific_suggestions.append(f"Tɩ bɩɩd t'a maan tɩ : {short[:30]}...")
                elif language == "di":
                    specific_suggestions.append(f"Ka o lajɛ : {short[:40]}...")
        
        # Combiner les suggestions
        all_suggestions = specific_suggestions + suggestions_dict.get(language, suggestions_dict["fr"])
        return list(set(all_suggestions))[:3]  # Éviter les doublons, limiter à 3

    # ------------------- OLLAMA -------------------
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Plus bas pour plus de cohérence
                "top_p": 0.8,
                "num_predict": 250,  # Plus court
                "num_ctx": 1024,
                "repeat_penalty": 1.1
            }
        }
        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}).get("content") or "").strip()
        except Exception:
            return ""

    # ------------------- FALLBACK CONVERSATIONNEL -------------------
    def _fallback_conversational_response(self, question: str, rag_results: List[Dict], language: str, category: str) -> Dict:
        """Fallback conversationnel amélioré"""
        
        # Essayer de construire une réponse basée sur le RAG
        main_points = []
        for r in rag_results[:2]:
            short = (r.get("reponse_courte") or "").strip()
            if short:
                main_points.append(short)
        
        if main_points:
            # Prendre le premier point comme réponse principale
            main_response = main_points[0]
            if len(main_response) > 200:
                main_response = main_response[:197] + "..."
        else:
            # Réponse générique
            fallback_responses = {
                "fr": "Je vais vous donner des informations utiles basées sur mes connaissances du Burkina Faso.",
                "mo": "M na maan tagmasg n bas Burkina Faso yel-wεεnẽ ye.",
                "di": "N bɛna kunnafoni di i ma minnu bɛ se ka i dɛmɛ Burkina Faso la."
            }
            main_response = fallback_responses.get(language, fallback_responses["fr"])
        
        # Générer des suggestions
        suggestions = self._generate_suggestions(rag_results, language)
        
        return {
            "reponse": main_response,
            "suggestions": suggestions,
            "mode": "structured_rag",
            "langue": language,
            "categorie": category,
            "sources_utilisees": len(rag_results),
            "timestamp": datetime.utcnow().isoformat(),
            "cache_hit": False
        }


# INSTANCE GLOBALE
ai_brain = AIBrain()