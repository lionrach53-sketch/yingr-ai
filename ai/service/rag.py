# ai/service/rag.py
import logging
import io
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from .vector_store import VectorStore
from .rag_enhancer import rag_enhancer
from .hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

# Singleton pour le modèle d'embedding (éviter chargements multiples)
_embedding_model_cache = None

def get_embedding_model():
    """Retourne le modèle d'embedding (singleton)"""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        logger.info("Chargement initial du modèle d'embedding...")
        _embedding_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Modèle d'embedding chargé: all-MiniLM-L6-v2")
    return _embedding_model_cache

class RAGService:
    """
    RAGService : Retrieval-Augmented Generation
    Utilise un index FAISS pour retrouver les documents pertinents
    et un modèle SentenceTransformer pour générer des embeddings.
    """
    def __init__(self):
        logger.info("Initialisation du RAGService...")
        self.embedding_model = get_embedding_model()
        self.vector_store = VectorStore(dim=384)
        logger.info("Index FAISS et métadonnées chargés avec succès.")

    # =========================
    # Méthodes publiques
    # =========================
    def ingest(self, texts: List[str], source: str):
        """
        Ajouter des textes/documents à l'index FAISS
        """
        embeddings = self.embed(texts)
        metadata = [{"source": source, "text": txt} for txt in texts]
        self.vector_store.add(embeddings, metadata)
        logger.info(f"{len(texts)} documents ingérés dans l'index.")

# ... [le reste du code reste identique] ...

    def ask(self, query: str, k: int = 5, language: str = None, category: str = None, min_confidence: float = 0.65) -> Tuple[str, str]:
        """
        Récupérer une réponse pertinente et le contexte
        Version OPTIMISÉE avec filtrage strict et formatage propre.
        """
        # Enrichissement de requête
        enriched_query = rag_enhancer.enrich_query(query, category)
        logger.info(f"📝 Requête enrichie: '{enriched_query[:100]}'")
        
        query_vector = self.embed([enriched_query])
        
        # Rechercher plus de résultats pour re-ranking
        search_k = k * 3 if (language or category) else k
        results, scores = self.vector_store.search(query_vector, k=search_k, return_scores=True)

        if not results:
            return self._get_fallback_response(language), ""
        
        # Convertir distance L2 en score de similarité (0-1)
        similarities = [1.0 / (1.0 + d) for d in scores]
        best_similarity = max(similarities) if similarities else 0.0
        logger.info(f"📊 Meilleure similarité: {best_similarity:.3f} (seuil: {min_confidence})")
        
        # 🔥 FILTRE CRITIQUE : rejeter si similarité trop faible
        if best_similarity < min_confidence:
            logger.warning(f"⚠️ Similarité trop faible ({best_similarity:.3f} < {min_confidence}) - Fallback activé.")
            return self._get_fallback_response(language), ""
        
        # Filtrer par langue
        if language:
            filtered_results = []
            filtered_similarities = []
            for idx, r in enumerate(results):
                source = r.get("source", "")
                lang_match = f"-{language}" in source
                if lang_match and similarities[idx] >= min_confidence:
                    filtered_results.append(r)
                    filtered_similarities.append(similarities[idx])
                if len(filtered_results) >= k:
                    break
            
            if filtered_results:
                results = filtered_results
                similarities = filtered_similarities
            else:
                logger.error(f"❌ Aucun résultat pour la langue {language} avec seuil {min_confidence}")
                return self._get_fallback_response(language), ""
        
        # Re-ranking hybride
        logger.info(f"🎯 Re-ranking hybride de {len(results)} résultats...")
        results, similarities = HybridSearch.rerank_results(
            query=query,
            results=results,
            semantic_scores=similarities,
            keyword_weight=0.5
        )
        
        # Prendre les k meilleurs
        results = results[:k]
        similarities = similarities[:k]
        
        # 🔥 FORMATAGE PROPRE DU CONTEXTE (CRITIQUE)
        # On prend UNIQUEMENT le meilleur résultat comme contexte principal
        if results:
            best_result = results[0]
            
            # Extraire les parties utiles de votre JSON
            text_content = best_result.get("text", "")
            context_for_llm = ""
            
            # Si le texte contient votre structure JSON parsée
            if "reponse_detaillee" in text_content or "REPONSE_DETAIL:" in text_content:
                # Format optimisé pour le LLM
                lines = text_content.split('\n')
                for line in lines:
                    if line.startswith("QUESTION:") or line.startswith("REPONSE_DETAIL:") or line.startswith("REPONSE_COURTE:"):
                        context_for_llm += line + "\n"
                    elif line.startswith("CONSEIL:") and line.strip() != "CONSEIL:":
                        context_for_llm += line + "\n"
            else:
                # Fallback : utiliser tout le texte mais limiter la longueur
                context_for_llm = text_content[:500] + "..." if len(text_content) > 500 else text_content
            
            # La réponse finale à retourner
            final_answer = ""
            if "reponse_courte" in text_content:
                # Extraire la réponse courte
                import re
                match = re.search(r"REPONSE_COURTE:\s*(.+)", text_content)
                if match:
                    final_answer = match.group(1).strip()
            
            if not final_answer:
                # Fallback : prendre le début du texte
                final_answer = text_content[:150].strip() + "..."
            
            logger.info(f"✅ Contexte formaté ({len(context_for_llm)} chars), similarité: {similarities[0]:.3f}")
            return final_answer, context_for_llm
        
        return self._get_fallback_response(language), ""
    
    def _get_fallback_response(self, language: str) -> str:
        """Réponses de fallback par langue"""
        fallbacks = {
            "fr": "Je n'ai pas d'information suffisamment précise sur ce point. Pouvez-vous reformuler ou poser une question sur un autre sujet ?",
            "mo": "M pa tara tagmasg sẽn yɩɩd sẽn na yɩlẽ f meng ye. F tõog n kãn-y a wa tɩ f sẽn dat n kẽ sabl fãa ?",
            "di": "N tɛ kunnafoni ɲɛman sɔr o kɔnɔ. Yala i bɛ se ka ɲininkali in labɛn wa, walima ɲininkali wɛrɛ ye wa ?"
        }
        return fallbacks.get(language, fallbacks["fr"])


        
        # 🔥 NOUVEAU : Re-ranking hybride (sémantique + mots-clés)
        logger.info(f"🎯 Re-ranking hybride de {len(results)} résultats...")
        results, similarities = HybridSearch.rerank_results(
            query=query,  # Question ORIGINALE (pas enrichie) pour les mots-clés
            results=results,
            semantic_scores=similarities,
            keyword_weight=0.5  # 50% mots-clés, 50% sémantique
        )
        logger.info(f"✅ Re-ranking terminé. Top score: {similarities[0]:.3f}")
        
        # Prendre les k meilleurs résultats après re-ranking
        results = results[:k]
        similarities = similarities[:k]
        
        # Extraire le texte des résultats, fusionner et limiter la répétition
        context_texts = []
        answer_only_texts = []  # SEULEMENT les réponses, AUCUNE question
        seen_texts = set()
        
        for r in results:
            txt = r.get("text", "")
            if txt and txt not in seen_texts:
                # Parser pour séparer question et réponse
                if "\n\n" in txt:
                    parts = txt.split("\n\n", 1)
                    if len(parts) >= 2:
                        question_part = parts[0].strip()
                        answer_part = parts[1].strip()
                        
                        # Context complet pour logs
                        context_texts.append(f"Q: {question_part}\nR: {answer_part}")
                        # Mais ENVOYER AU LLM SEULEMENT LA RÉPONSE!
                        answer_only_texts.append(answer_part)
                        seen_texts.add(txt)
                    else:
                        context_texts.append(txt)
                        answer_only_texts.append(txt)
                        seen_texts.add(txt)
                else:
                    # Pas de séparation Q/R claire
                    context_texts.append(txt)
                    answer_only_texts.append(txt)
                    seen_texts.add(txt)

        # IMPORTANT: Le contexte envoyé au LLM contient SEULEMENT les réponses
        # Pas de questions pour éviter confusion!
        context = "\n\n---\n\n".join(answer_only_texts)
        
        # Sélectionner la réponse la plus pertinente (premier résultat)
        if answer_only_texts:
            most_relevant = answer_only_texts[0]
            return most_relevant.strip(), context
        
        return "Aucune information pertinente trouvée.", ""

    # =========================
    # Analyse PDF / Documents
    # =========================
    def ingest_pdf(self, pdf_bytes: bytes, source: str):
        """
        Convertir un PDF en texte et ingérer
        """
        try:
            import fitz  # PyMuPDF  # type: ignore
        except ImportError:
            raise ImportError("PyMuPDF est requis pour ingérer des PDF: pip install pymupdf")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = [page.get_text() for page in doc]
        self.ingest(texts, source)
        logger.info(f"PDF ingéré avec {len(texts)} pages.")

    # =========================
    # Analyse Images
    # =========================
    def ingest_image(self, image_bytes: bytes, source: str):
        """
        Analyse sommaire d'image (OCR ou description simple)
        """
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except ImportError:
            raise ImportError("PIL et pytesseract sont requis pour ingérer des images: pip install pillow pytesseract")

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        if text.strip():
            self.ingest([text], source)
            logger.info(f"Texte extrait de l'image et ingéré.")
        else:
            logger.warning("Aucun texte détecté dans l'image.")

    # =========================
    # Embedding
    # =========================
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Retourne les embeddings pour une liste de textes
        """
        return self.embedding_model.encode(texts, convert_to_numpy=True)
    
    def _enrich_query(self, query: str, category: str = None) -> str:
        """
        Enrichit la question avec des mots-clés spécifiques à la catégorie
        pour améliorer la recherche sémantique.
        N'enrichit QUE si la question est assez longue (> 6 mots) pour éviter la dilution.
        """
        if not category:
            return query
        
        # Ne pas enrichir les courtes questions pour éviter la dilution
        word_count = len(query.split())
        if word_count < 4:  # Questions très courtes: pas d'enrichissement
            return query
            
        # Mapping des catégories vers des mots-clés pertinents
        category_keywords = {
            "plantes medicinales": "plante médicale santé remède",
            "plantesmedicinales": "plante médicale santé remède",
            "transformation pfnl": "transformation karité noix produit",
            "transformationpfnl": "transformation karité noix produit",
            "science pratique - saponification": "savon fabrication soude huile",
            "sciencepratiquesaponification": "savon fabrication soude huile",
            "metiers informels": "métier travail informel secteur",
            "metiersinformels": "métier travail informel secteur",
            "civisme": "citoyen devoir responsabilité",
            "spiritualite et traditions": "tradition spirituelle culture",
            "spiritualiteettraditions": "tradition spirituelle culture",
            "developpement personnel": "compétence développement objectif",
            "developpementpersonnel": "compétence développement objectif",
            "mathematiques pratiques": "calcul mathématique surface",
            "mathematiquespratiques": "calcul mathématique surface",
            "general": "",  # Pas d'enrichissement pour general
        }
        
        # Normaliser la catégorie pour la recherche
        import unicodedata
        normalized_cat = category.lower()
        normalized_cat = unicodedata.normalize('NFD', normalized_cat)
        normalized_cat = ''.join(c for c in normalized_cat if unicodedata.category(c) != 'Mn')
        normalized_cat = normalized_cat.replace(' ', '').replace('&', '').replace('-', '')
        
        keywords = category_keywords.get(normalized_cat, "")
        
        # Ajouter les mots-clés à la fin de la question
        if keywords and word_count >= 4:
            logger.info(f"🔍 Question enrichie: '{query}' + '{keywords}' (catégorie: {category})")
            return f"{query} {keywords}"
        return query
