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

    def ask(self, query: str, k: int = 5, language: str = None, category: str = None, min_confidence: float = 0.40) -> Tuple[str, str]:
        """
        Récupérer une réponse pertinente et le contexte
        Filtre par langue et catégorie si spécifiés
        min_confidence: seuil de similarité (0-1). Plus bas = plus permissif. Défaut 0.40
        
        AMÉLIORÉ avec enrichissement de requête et re-ranking hybride
        """
        # 🔥 NOUVEAU : Enrichir la question avec synonymes et contexte
        enriched_query = rag_enhancer.enrich_query(query, category)
        logger.info(f"📝 Requête enrichie: '{enriched_query[:100]}'")
        
        query_vector = self.embed([enriched_query])
        
        # Rechercher plus de résultats pour re-ranking
        search_k = k * 3 if (language or category) else k
        results, scores = self.vector_store.search(query_vector, k=search_k, return_scores=True)

        if not results:
            return "Je n'ai pas trouvé d'information sur ce sujet. Pourriez-vous reformuler votre question ?", ""
        
        # Convertir distance L2 en score de similarité (0-1)
        # Distance L2: 0 = identique, plus grand = plus différent
        # On normalise: similarité = 1 / (1 + distance)
        similarities = [1.0 / (1.0 + d) for d in scores]
        
        # Vérifier si le meilleur résultat dépasse le seuil
        best_similarity = max(similarities) if similarities else 0.0
        logger.info(f"📊 Meilleure similarité: {best_similarity:.3f} (seuil: {min_confidence})")
        
        if best_similarity < min_confidence:
            logger.warning(f"❌ Similarité trop faible ({best_similarity:.3f} < {min_confidence})")
            return "Je ne suis pas sûr de comprendre votre question. Pourriez-vous la reformuler ou choisir un sujet parmi les catégories disponibles ?", ""

        # Filtrer par langue ET catégorie si spécifiées, en gardant les scores
        # ⚠️ IMPORTANT: Si category='general', on filtre SEULEMENT par langue (pas de filtre catégorie)
        if language:
            filtered_results = []
            filtered_scores = []
            for idx, r in enumerate(results):
                source = r.get("source", "")
                lang_match = f"-{language}" in source
                if lang_match:
                    filtered_results.append(r)
                    filtered_scores.append(similarities[idx] if idx < len(similarities) else 0.0)
                if len(filtered_results) >= k:
                    break
            if len(filtered_results) == 0:
                logger.error(f"❌ Aucun résultat pour la langue {language}")
                return "Je n'ai pas trouvé d'information sur ce sujet dans cette langue. Pourriez-vous reformuler votre question ?", ""
            else:
                logger.info(f"✅ {len(filtered_results)} résultats trouvés pour la langue {language}")
                results = filtered_results
                similarities = filtered_scores
        
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
