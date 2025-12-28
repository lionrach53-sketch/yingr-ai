# ai/service/conversation.py
"""
Service de conversation intelligent avec détection de langue et analyse contextuelle
"""
import logging
import re
from typing import Tuple, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationService:
    """
    Service de conversation intelligent qui :
    - Détecte la langue (français, mooré, dioula)
    - Analyse l'intention (salutation, question, demande d'aide)
    - Génère des réponses contextuelles
    - Pose des questions de clarification si nécessaire
    """
    
    def __init__(self):
        # Patterns de salutations par langue
        self.greetings = {
            'fr': ['bonjour', 'salut', 'bonsoir', 'hello', 'hi', 'coucou', 'hey'],
            'mo': ['ne y kɔɔrɛ', 'ne y kyɛɛrɛ', 'ne y zɔɔrɛ', 'woto', 'an-soama'],
            'di': ['i ni sɔgɔma', 'i ni tile', 'i ni wula', 'aw ni ce']
        }
        
        # Patterns de remerciements
        self.thanks = {
            'fr': ['merci', 'thank', 'grand merci', "c'est gentil", 'ok merci'],
            'mo': ['barka', 'yamba', 'n barika', 'la fii'],
            'di': ['i ni ce', 'i ni ɲininka', 'an bi se']
        }
        
        # ✅ NOUVEAU : Patterns pour questions sur l'identité de l'IA
        self.identity_questions = {
            'fr': [
                'quel est ton nom', 'comment tu t\'appelles', 'qui es-tu', 
                'tu es qui', 'ton nom', 'vous êtes', 'tu es', 'présente toi',
                'c\'est quoi ton nom', 'tu t\'appelles comment', 'ton identité',
                'qui es tu', 'qui est-tu', 'quel est votre nom', 'comment vous appelez-vous'
            ],
            'mo': [
                'fo yembr yaa bo', 'fo sẽn get fo yembr yaa bo', 'fo yaa bo',
                'fo yembr', 'tɩ tʋgd fo', 'fo yaa', 'yembr fo', 'tɩ maan fo yɩɩlã',
                'fo sẽn yaa bo', 'fo yɩɩl yaa bo', 'fo yɩɩ bo', 'fo tʋmde yaa bo'
            ],
            'di': [
                'i tɔgɔ ye di', 'i ye jɔn ye', 'i tɔgɔ bɛ di', 
                'i yɛrɛ', 'i ye mun ye', 'i tɔgɔ', 'i bɛ di',
                'i tɔgɔ min ye', 'i ka tɔgɔ di', 'i ka jɔli di',
                'i bɛ jɔn ye', 'i ka dɔnko di'
            ]
        }
        
        # Patterns d'affirmation/satisfaction
        self.affirmations = {
            'fr': ['oui', 'ok', 'bien', 'compris', 'parfait', "d'accord", 'exact'],
            'mo': ['eeŋ', 'aaŋ', 'awã', 'n bãng', 'raabo'],
            'di': ['ɔ̃w', 'awɔ', 'tiɲɛ', 'a ka ɲi']
        }
        
        # ✅ NOUVEAU : Patterns d'au revoir
        self.goodbyes = {
            'fr': ['au revoir', 'bye', 'à plus', 'ciao', 'adieu', 'bonne journée', 'à bientôt'],
            'mo': ['wɩnd ne y taabo', 'y taare', 'ne y windga', 'wɩnd n yɩɩs ne fo', 'kɩnd ne fo'],
            'di': ['an bɛn kɔfɛ', 'i ni ce', 'i ni su', 'o la fɔlɔ', 'a bɛn kɔfɛ']
        }
        
        # Patterns de questions simples (oui/non, confirmation)
        self.simple_responses = {
            'fr': ['ok', 'd\'accord', 'compris', 'entendu', 'super', 'génial', 'parfait', 'cool', 'merci'],
            'mo': ['yɩɩ sõma', 'n bãng', 'raabo', 'a yɩɩ', 'barka'],
            'di': ['a ka ɲi', 'n y\'a faamu', 'a bɛ kɛ', 'i ni ce']
        }
        
        # Mots-clés par langue pour détection
        self.lang_markers = {
            'fr': ['est', 'le', 'la', 'les', 'un', 'une', 'des', 'que', 'qui', 'comment', 'pourquoi', 'quand'],
            'mo': ['yɩlɩg', 'woto', 'yaa', 'ne', 'sãn', 'kẽ', 'n', 'na', 'bɩ', 'pʋgẽ', 'taaba'],
            'di': ['ye', 'ka', 'bɛ', 'kɛ', 'ni', 'ma', 'wa', 'kɔrɔ', 'fɔ', 'min', 'tɛ']
        }
        
        # Questions types par catégorie
        self.follow_up_questions = {
            'histoire': {
                'fr': "Voulez-vous en savoir plus sur l'histoire du Burkina Faso, ses personnalités ou ses événements importants ?",
                'mo': "Y bãng n ka Burkina Faso tarek, n taaba yamb ned n sã n kẽnd be kɔɔga ?",
                'di': "I b'a fɛ ka Burkina Faso tariku, a ka mɔgɔba walima a ka fɛn kunba ye wa ?"
            },
            'agriculture': {
                'fr': "Souhaitez-vous des informations sur les cultures, les techniques agricoles ou les saisons de plantation ?",
                'mo': "Y bãng n ka bʋʋlg tɩɩsa, bãnd tigsi ned bãnd yĩnga kɔɔga ?",
                'di': "I b'a fɛ ka sɛnɛkɛ kow, sɛnɛkɛli kow walima donkow ye wa ?"
            },
            'sante': {
                'fr': "Avez-vous besoin d'informations sur une maladie spécifique, la prévention ou les remèdes traditionnels ?",
                'mo': "Y bãng kɩndɩg tɩɩsa, kɩndɩg yɩlsgo ned tãab tɩɩm kɔɔga ?",
                'di': "I b'a fɛ ka bana dɔ ye, bana tanga walima fura kow ye wa ?"
            },
            'general': {
                'fr': "Comment puis-je vous aider aujourd'hui ? Vous avez des questions sur l'agriculture, la santé, l'histoire, ou autre chose ?",
                'mo': "Woto n tõe yɩɩlã yem bo ? Y kẽ kɩtugã bãndã, kɩndɩgã, tarekã ned tʋʋma be sãn ?",
                'di': "Ne bɛ se ka i dɛmɛ cogo di bi ? I ka ɲininka b'i fɛ sɛnɛkɛ, kɛnɛya, tariku walima fɛn wɛrɛ kan wa ?"
            }
        }
        
        # Réponses aux salutations
        self.greeting_responses = {
            'fr': [
                "Bonjour ! Je suis l'IA Souveraine du Burkina Faso. Comment puis-je vous aider aujourd'hui ?",
                "Salut ! Ravi de vous parler. Que voulez-vous savoir ?",
                "Bonjour ! Je suis là pour répondre à vos questions sur le Burkina Faso. Que cherchez-vous ?"
            ],
            'mo': [
                "Ne y kɔɔrɛ ! M yaa Burkina Faso AI taaba. Woto n tõe yɩɩlã yem bo ?",
                "An-soama ! N yaa yõodo n yɩ ne. Fo sãn ye ?",
                "Waka ! M yaa yãnd b'a yɩ ne Burkina Faso sũur. Fo kẽ be kɩtugã ?"
            ],
            'di': [
                "I ni sɔgɔma ! Ne ye Burkina Faso AI ye. Ne bɛ se ka i dɛmɛ cogo di ?",
                "I ni ce ! Ne b'a fɛ ka kuma ni i ye. I b'a fɛ ka mun lɔn ?",
                "I ka kɛnɛ ! Ne ye yan ka i ɲininkaw jaabi. I be mun ɲini ?"
            ]
        }
        
        # ✅ NOUVEAU : Réponses aux questions d'identité
        self.identity_responses = {
            'fr': [
                "Je suis **YINGR-AI**, une intelligence artificielle souveraine dédiée au Burkina Faso. "
                "Mon nom signifie « Intelligence » en mooré (YINGR) combiné avec l'intelligence artificielle (AI). "
                "Je suis ici pour vous aider avec des informations sur l'agriculture, la santé, l'éducation, "
                "la culture et bien d'autres sujets concernant le Burkina Faso.\n\n"
                "Je fonctionne avec une technologie de RAG (Recherche Augmentée par Génération) qui me permet "
                "de m'appuyer sur une base de connaissances fiables tout en ayant des capacités de raisonnement. "
                "Je peux aussi vous répondre en mooré et en dioula !\n\n"
                "Comment puis-je vous aider aujourd'hui ?",
                
                "**YINGR-AI** à votre service ! Je suis l'assistant IA souverain du Burkina Faso. "
                "YINGR signifie « Intelligence » en mooré, et AI c'est pour Intelligence Artificielle. "
                "Je suis conçu pour vous fournir des informations précises et utiles sur notre cher pays.\n\n"
                "Je peux vous parler d'agriculture, de santé, d'éducation, de culture, d'histoire, "
                "d'économie, et bien plus encore. Je comprends et parle français, mooré et dioula !\n\n"
                "Que souhaitez-vous savoir ?"
            ],
            'mo': [
                "**YINGR-AI** la mam yaa. YINGR yɩɩd bʋʋm-yelẽ ye Mooré pʋgẽ la AI yaa bool-nonglem ye. "
                "Mam na yɩll n yaa Burkina Faso bool-nonglem soaba. Mam tõe n kɩt f meng n bas tʋʋm-noogo, "
                "koongo, bʋʋm-yelẽ, kũun, la yel-wεεn wã fãa sẽn gɩdg Burkina Faso pʋgẽ.\n\n"
                "Mam tʋmda tɩ yaa RAG (Recherche Augmentée par Génération) sʋka. Bʋɩl-woto tõog n maan tɩ "
                "mam tara tõnd tagmasg n karengr sẽn tɩ yɩɩ n yɩɩme n yãag la mam tara bʋʋm-yelẽ nonglem. "
                "Mam tõe n kãn-wẽng Moorẽ, Dioula la Fãransẽ pʋgẽ !\n\n"
                "Tõnd nonglem maana yaa ?",
                
                "Mam yaa **YINGR-AI**, Burkina Faso bool-nonglem soaba. YINGR yɩɩd bʋʋm-yelẽ ye, "
                "AI yaa bool-nonglem ye. Mam na yɩll ne fo ye tɩ kɩt yel-wεεn sẽn be Burkina Faso pʋgẽ.\n\n"
                "Mam tõe n kɩt yel-wεεn n bas sɛnɛ, koongo, ladob-tʋʋm, kũun, tarek, la yel-wεεn wã fãa. "
                "Mam tara bʋʋm n gʋls Moorẽ, Dioula la Fãransẽ pʋgẽ.\n\n"
                "Fo kẽ be kɩtugã ?"
            ],
            'di': [
                "N ye **YINGR-AI** ye. YINGR bɛ kɔrɔfɛ kan na, o bɛ kuma « Ladɔnni » la, AI bɛ kuma « Bool-nonglem » ye. "
                "N yɛrɛ yɛrɛ bɛ Burukina Faso dɛmɛbaga ye. N bɛ se ka i dɛmɛ kɔrɔw, kɔrɔfɛ, ladɔnni, laɲini, "
                "ani fɛn wɛrɛw fɛ minnu bɛ Burukina Faso la.\n\n"
                "N bɛ baara kɛ RAG (Recherche Augmentée par Génération) ye. O bɛ kɛ cogo min na, n bɛ se ka "
                "kunnafoni siratigi sɔr n'o fɛ n'o fɛ, n bɛ fɛn wɛrɛw fɔ ka ɲɛ. N bɛ se ka dioula, mooré ani "
                "faransi kan fɔ !\n\n"
                "N bɛ se ka i dɛmɛ di cogo jumɛn na di ?",
                
                "**YINGR-AI** n yɛrɛ ye. YINGR bɛ sɔrɔ mooré kan na, o bɛ kɔrɔfɛ « Ladɔnni » fɔ, AI bɛ « Bool-nonglem » fɔ. "
                "N bɛ Burukina Faso dɛmɛbaga ye. N bɛ i dɛmɛ sɛnɛkɛ, kɛnɛya, ladɔnni, laɲini, tariku, "
                "waria, ani fɛn camanw fɛ.\n\n"
                "N bɛ kunnafoni siratigiw sɔr n'o fɛ n'o fɛ, n bɛ fɛn wɛrɛw fɔ ka ɲɛ. N bɛ dioula, mooré ani "
                "faransi kan fɔ.\n\n"
                "I b'a fɛ ka mun lɔn ?"
            ]
        }
        
        # ✅ NOUVEAU : Réponses d'au revoir
        self.goodbye_responses = {
            'fr': [
                "Au revoir ! Merci d'avoir utilisé YINGR-AI. À bientôt pour de nouvelles conversations sur le Burkina Faso !",
                "À bientôt ! N'hésitez pas à revenir si vous avez d'autres questions. Bonne journée !",
                "Au revoir et merci ! Je reste à votre disposition pour toute question sur le Burkina Faso."
            ],
            'mo': [
                "Wɩnd ne y taabo ! Barka sẽn yɩɩ n tʋm YINGR-AI ye. Tɩ seng fo lebg ne tõnd ye n kɩt Burkina Faso yel-wεεnẽ.",
                "Y taare ! Fo sẽn tõog n lebg ye, fo tɩ n yel. N yɩɩs ne fo !",
                "Wɩnd n yɩɩs ne fo ! Barka, la mam be yemb ne fo ye n bas Burkina Faso yel-wεεnẽ."
            ],
            'di': [
                "An bɛn kɔfɛ ! I ni ce ka YINGR-AI baara kɛ. An bɛna segin kumakan wɛrɛw kɛ Burukina Faso la !",
                "I ni su ! N'i bɛ ɲininka wɛrɛ sɔrɔ, i k'a fɔ ne ye. I ni tile !",
                "O la fɔlɔ ! N bɛ se ka i dɛmɛ kɔfɛ, n'i bɛ ɲininka wɛrɛw sɔrɔ Burukina Faso la."
            ]
        }
        
        # ✅ NOUVEAU : Réponses simples (ok, merci, compris)
        self.simple_response_texts = {
            'fr': [
                "👍 Parfait ! Souhaitez-vous approfondir ce sujet ou passer à autre chose ?",
                "✅ Compris ! Voulez-vous continuer sur ce sujet ou avez-vous une autre question ?",
                "👌 D'accord ! Je suis là si vous avez besoin de plus d'informations."
            ],
            'mo': [
                "👍 Yɩɩ sõma ! Fo sẽn tõog n bas tɩ yel woto wa tɩ tʋm yel wɛɛngẽ ?",
                "✅ N bãng ! Fo sẽn tõog n bas tɩ yel woto wa tɩ kɩt yel wɛɛngẽ ?",
                "👌 Raabo ! Mam be yemb ne fo ye n'i tara tagmasg wɛɛngẽ."
            ],
            'di': [
                "👍 A ka ɲi ! Yala i b'a fɛ ka kuma in jigin wa, walima kuma wɛrɛw la ?",
                "✅ N y'a faamu ! I b'a fɛ ka o lajɛ wa, walima i bɛ ɲininka wɛrɛ sɔrɔ ?",
                "👌 A bɛ kɛ ! N be yan n'i bɛ kunnafoni wɛrɛw fɛ."
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """
        Détecte la langue du texte (fr, mo, di)
        """
        import re
        
        text_lower = text.lower()
        scores = {'fr': 0, 'mo': 0, 'di': 0}
        
        # Compter les marqueurs de langue avec word boundaries
        for lang, markers in self.lang_markers.items():
            for marker in markers:
                # Utiliser word boundary pour éviter les faux positifs
                # \b ne marche pas avec les caractères spéciaux, alors on cherche avec espaces/ponctuation
                pattern = r'(?:^|\s|[,;.!?])' + re.escape(marker) + r'(?:\s|[,;.!?]|$)'
                if re.search(pattern, text_lower):
                    scores[lang] += 1
        
        # Vérifier les caractères spéciaux mooré et dioula
        if any(char in text for char in ['ɩ', 'ɛ', 'ɔ', 'ʋ', 'ɲ', 'ŋ']):
            if 'ɩ' in text or 'ʋ' in text or 'ɛ' in text:
                scores['mo'] += 3
            if 'ɔ' in text or 'ɲ' in text:
                scores['di'] += 2
        
        # Retourner la langue avec le score le plus élevé
        detected = max(scores, key=scores.get)
        
        # Si aucun marqueur, par défaut français
        if scores[detected] == 0:
            return 'fr'
        
        logger.info(f"🌍 Langue détectée: {detected} (scores: {scores})")
        return detected
    
    def detect_intent(self, text: str, lang: str) -> str:
        """
        Détection améliorée des intentions :
        - identity: question sur l'identité de l'IA
        - greeting: salutation
        - thanks: remerciement
        - goodbye: au revoir
        - simple: réponse simple (ok, merci, compris)
        - affirmation: confirmation
        - question: question
        - statement: déclaration
        """
        text_lower = text.lower().strip()
        
        # ✅ 1. Vérifier question sur l'identité (priorité haute)
        if lang in self.identity_questions:
            for marker in self.identity_questions[lang]:
                if marker in text_lower:
                    logger.info(f"🎯 Intention détectée: identity (marqueur: '{marker}')")
                    return 'identity'
        
        # ✅ 2. Vérifier au revoir
        if lang in self.goodbyes:
            for marker in self.goodbyes[lang]:
                if marker in text_lower:
                    logger.info(f"🎯 Intention détectée: goodbye (marqueur: '{marker}')")
                    return 'goodbye'
        
        # 3. Vérifier salutation
        if lang in self.greetings:
            for greet in self.greetings[lang]:
                if greet in text_lower:
                    logger.info(f"🎯 Intention détectée: greeting (marqueur: '{greet}')")
                    return 'greeting'
        
        # 4. Vérifier remerciement
        if lang in self.thanks:
            for thank in self.thanks[lang]:
                if thank in text_lower:
                    logger.info(f"🎯 Intention détectée: thanks (marqueur: '{thank}')")
                    return 'thanks'
        
        # ✅ 5. Vérifier réponse simple (ok, merci, compris, etc.)
        if lang in self.simple_responses:
            for simple in self.simple_responses[lang]:
                if simple in text_lower and len(text_lower.split()) <= 3:
                    logger.info(f"🎯 Intention détectée: simple (marqueur: '{simple}')")
                    return 'simple'
        
        # 6. Vérifier affirmation
        if lang in self.affirmations:
            for affirm in self.affirmations[lang]:
                if affirm in text_lower:
                    logger.info(f"🎯 Intention détectée: affirmation (marqueur: '{affirm}')")
                    return 'affirmation'
        
        # 7. Vérifier si c'est une question
        question_markers = {
            'fr': ['?', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'que', 'quel', 'quelle', 'est-ce que', 'qu\'est-ce que'],
            'mo': ['?', 'woto', 'yaa', 'fo', 'ãnsɛɛm', 'kãn', 'bɩ', 'sãn', 'ned'],
            'di': ['?', 'mun', 'cogo di', 'joli', 'yan', 'min', 'dɔ', 'jɔn', 'dɔɔni']
        }
        
        if lang in question_markers:
            for marker in question_markers[lang]:
                if marker in text_lower:
                    logger.info(f"🎯 Intention détectée: question (marqueur: '{marker}')")
                    return 'question'
        
        logger.info(f"🎯 Intention détectée: statement (par défaut)")
        return 'statement'
    
    def generate_greeting_response(self, lang: str) -> str:
        """Génère une réponse de salutation"""
        import random
        responses = self.greeting_responses.get(lang, self.greeting_responses['fr'])
        return random.choice(responses)
    
    def generate_thanks_response(self, lang: str) -> str:
        """Génère une réponse aux remerciements"""
        responses = {
            'fr': "De rien ! N'hésitez pas si vous avez d'autres questions. 😊",
            'mo': "Bãmb ra ! Fo kẽ kɩtugã be, fo tɩ n yel.",
            'di': "A tɛ fɔ ! N'i bɛ ɲininka wɛrɛ, i k'a fɔ ne ye."
        }
        return responses.get(lang, responses['fr'])
    
    # ✅ NOUVELLE MÉTHODE : Générer réponse d'identité
    def generate_identity_response(self, lang: str) -> str:
        """Génère une réponse pour présenter l'IA"""
        import random
        responses = self.identity_responses.get(lang, self.identity_responses['fr'])
        return random.choice(responses)
    
    # ✅ NOUVELLE MÉTHODE : Générer réponse d'au revoir
    def generate_goodbye_response(self, lang: str) -> str:
        """Génère une réponse d'au revoir"""
        import random
        responses = self.goodbye_responses.get(lang, self.goodbye_responses['fr'])
        return random.choice(responses)
    
    # ✅ NOUVELLE MÉTHODE : Générer réponse simple
    def generate_simple_response(self, lang: str) -> str:
        """Génère une réponse simple (ok, merci, compris)"""
        import random
        responses = self.simple_response_texts.get(lang, self.simple_response_texts['fr'])
        return random.choice(responses)
    
    def suggest_follow_up(self, category: str, lang: str) -> str:
        """Suggère une question de suivi selon la catégorie"""
        # Toujours retourner une question générale car nous avons de nouvelles catégories
        # qui ne sont pas dans le dictionnaire follow_up_questions
        responses = {
            'fr': f"Avez-vous d'autres questions sur {category} ou un autre sujet ?",
            'mo': f"Y kẽ kɩtugã be {category} ned tʋʋma be sãn ?",
            'di': f"I ka ɲininka wɛrɛw b'i fɛ {category} walima fɛn wɛrɛ kan wa ?"
        }
        return responses.get(lang, responses['fr'])
    
    def is_too_vague(self, text: str) -> bool:
        """Détermine si la question est trop vague"""
        words = text.lower().split()
        
        # Questions d'un ou deux mots sont généralement vagues
        if len(words) <= 2:
            return True
        
        # Patterns vagues
        vague_patterns = [
            r'^(quoi|comment|pourquoi|qui|que)\s*$',
            r'^(mun|woto|yaa)\s*$',
            r'^(aide|help|info)\s*$',
        ]
        
        return any(re.match(pattern, text.lower().strip()) for pattern in vague_patterns)
    
    def format_response(self, raw_answer: str, lang: str, intent: str, category: str, add_follow_up: bool = True) -> str:
        """
        Formate la réponse de manière conversationnelle
        IMPORTANT: Force la langue de la réponse selon la langue détectée
        """
        # Nettoyer la réponse brute
        answer = raw_answer.strip()
        
        # Retirer les préfixes génériques
        prefixes_to_remove = [
            "Selon les connaissances locales :",
            "Selon les connaissances locales: ",
            "D'après les informations :",
            "Voici ce que je sais :"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Si c'est une salutation, retourner juste la salutation
        if intent == 'greeting':
            return self.generate_greeting_response(lang)
        
        # Si c'est un remerciement
        if intent == 'thanks':
            return self.generate_thanks_response(lang)
        
        # ✅ Si c'est une question d'identité
        if intent == 'identity':
            return self.generate_identity_response(lang)
        
        # ✅ Si c'est un au revoir
        if intent == 'goodbye':
            return self.generate_goodbye_response(lang)
        
        # ✅ Si c'est une réponse simple
        if intent == 'simple':
            return self.generate_simple_response(lang)
        
        # VÉRIFIER SI LA RÉPONSE EST DANS LA MAUVAISE LANGUE
        # Si question en français mais réponse contient caractères mooré/dioula
        answer_lang = self.detect_language(answer)
        
        if lang != answer_lang:
            # La réponse est dans une mauvaise langue
            # Ajouter un message d'excuse dans la langue de l'utilisateur
            excuse_messages = {
                'fr': "⚠️ Désolé, la réponse disponible est en {detected_lang}. Voici ce que j'ai trouvé :\n\n",
                'mo': "⚠️ Gʋlsã, n gom sã n ka {detected_lang} ne. N ka yaa ne :\n\n",
                'di': "⚠️ Hakɛto, jaabi ye {detected_lang} la. Yan ne ye ne y'a sɔrɔ :\n\n"
            }
            
            lang_names = {'fr': 'français', 'mo': 'mooré', 'di': 'dioula'}
            excuse = excuse_messages.get(lang, excuse_messages['fr'])
            excuse = excuse.replace('{detected_lang}', lang_names.get(answer_lang, answer_lang))
            answer = excuse + answer
        
        # Pour les questions, formater la réponse
        formatted = answer
        
        # Ajouter une question de suivi si pertinent
        if add_follow_up and intent == 'question' and len(answer) > 50:
            follow_up = self.suggest_follow_up(category, lang)
            formatted = f"{answer}\n\n💡 {follow_up}"
        
        return formatted
    
    def analyze_and_respond(self, user_message: str, raw_rag_answer: str, category: str = "general") -> Dict[str, any]:
        """
        Analyse complète du message et génération de réponse intelligente
        
        Returns:
            Dict avec:
            - language: langue détectée
            - intent: intention (greeting, question, etc.)
            - response: réponse formatée
            - needs_clarification: bool si besoin de clarification
            - follow_up_suggestion: suggestion de question de suivi
        """
        # 1. Détection de langue
        lang = self.detect_language(user_message)
        
        # 2. Détection d'intention
        intent = self.detect_intent(user_message, lang)
        
        # 3. Vérifier si la question est trop vague
        needs_clarification = self.is_too_vague(user_message)
        
        # 4. Formater la réponse selon l'intention
        if intent == 'greeting':
            response = self.generate_greeting_response(lang)
            add_follow_up = True
        elif intent == 'thanks':
            response = self.generate_thanks_response(lang)
            add_follow_up = False
        elif intent == 'identity':
            response = self.generate_identity_response(lang)
            add_follow_up = True
        elif intent == 'goodbye':
            response = self.generate_goodbye_response(lang)
            add_follow_up = False
        elif intent == 'simple':
            response = self.generate_simple_response(lang)
            add_follow_up = True
        elif needs_clarification:
            clarification = {
                'fr': f"Je comprends que vous cherchez des informations, mais pourriez-vous être plus précis ? {self.suggest_follow_up(category, lang)}",
                'mo': f"N gom sã y kẽ kɩtugã, bala y tõe maan yɩɩlã sũuri ? {self.suggest_follow_up(category, lang)}",
                'di': f"Ne y'a faamu i b'a ɲini, nka i bɛ se k'a jira ka tɛmɛ wa ? {self.suggest_follow_up(category, lang)}"
            }
            response = clarification.get(lang, clarification['fr'])
            add_follow_up = False
        else:
            response = self.format_response(raw_rag_answer, lang, intent, category, add_follow_up=True)
            add_follow_up = False  # Déjà ajouté dans format_response
        
        # 5. Retourner l'analyse complète
        return {
            'language': lang,
            'intent': intent,
            'response': response,
            'needs_clarification': needs_clarification,
            'follow_up_suggestion': self.suggest_follow_up(category, lang) if add_follow_up else None
        }