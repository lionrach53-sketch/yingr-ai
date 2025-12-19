"""
RESOLVE HUB - Backend API avec Computer Vision LOCALE
IA 100% locale pour analyse de photos (maladies plantes)
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import hashlib
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resolvehub.db")

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models (mêmes que avant)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_anonymized = Column(Boolean, default=False)

class Expert(Base):
    __tablename__ = "experts"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    specialization = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    expert_id = Column(Integer, nullable=True)
    category = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    status = Column(String, default="open")
    ai_confidence_score = Column(Float, nullable=True)
    ai_extracted_keywords = Column(String, nullable=True)
    ai_photo_analysis = Column(Text, nullable=True)  # NOUVEAU
    photo_path = Column(String, nullable=True)  # NOUVEAU
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, nullable=False)
    sender_type = Column(String, nullable=False)
    sender_id = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    channel = Column(String, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

# Pydantic schemas
class MessageCreate(BaseModel):
    content: str
    phone_number: str
    channel: str = "app"
    photo_base64: Optional[str] = None  # NOUVEAU

class ExpertLogin(BaseModel):
    email: str
    password: str

# FastAPI app
app = FastAPI(title="RESOLVE HUB API - IA Locale", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

# ==========================================
# MODULE IA PHOTO LOCALE (Computer Vision)
# ==========================================

class LocalComputerVision:
    """
    Système de Computer Vision 100% LOCAL
    Détection de maladies des plantes sans API externe
    """
    
    def __init__(self):
        # Base de connaissances des maladies courantes au Burkina Faso
        self.diseases_database = {
            "mais_taches_jaunes": {
                "name": "Carence en Azote",
                "confidence_keywords": ["jaune", "feuille", "maïs", "sécher"],
                "symptoms": ["Jaunissement des feuilles du bas vers le haut", "Croissance ralentie"],
                "treatment": "Appliquer engrais NPK (10-10-10) à 50kg/ha. Améliorer drainage. Arrosage régulier matin/soir.",
                "urgency": "medium",
                "prevention": "Rotation des cultures, compost organique, analyse sol annuelle"
            },
            "mais_rouille": {
                "name": "Rouille du Maïs",
                "confidence_keywords": ["tache", "orange", "rouille", "poudre"],
                "symptoms": ["Pustules orange/brunes sur feuilles", "Aspect poudreuse"],
                "treatment": "Fongicide naturel (purin d'ortie dilué 1:10). Retirer feuilles infectées. Espacer plants.",
                "urgency": "high",
                "prevention": "Variétés résistantes, rotation, bon espacement"
            },
            "tomate_mildiou": {
                "name": "Mildiou de la Tomate",
                "confidence_keywords": ["tache", "brun", "noir", "tomate", "pourrir"],
                "symptoms": ["Taches brunes/noires sur feuilles", "Fruits pourrissent"],
                "treatment": "URGENT: Retirer plants infectés. Bouillie bordelaise. Éviter arrosage feuilles.",
                "urgency": "high",
                "prevention": "Paillage, arrosage au pied, aération"
            },
            "sorgho_charbon": {
                "name": "Charbon du Sorgho",
                "confidence_keywords": ["noir", "poudre", "épi", "sorgho"],
                "symptoms": ["Masse noire poudreuse remplace grains"],
                "treatment": "Détruire plants infectés (brûler). Traiter semences. Rotation 3 ans.",
                "urgency": "high",
                "prevention": "Semences certifiées traitées, rotation cultures"
            },
            "manioc_mosaique": {
                "name": "Mosaïque du Manioc",
                "confidence_keywords": ["mosaïque", "déformation", "feuille", "manioc"],
                "symptoms": ["Motif mosaïque jaune/vert sur feuilles", "Déformation"],
                "treatment": "Pas de traitement. Arracher et détruire. Utiliser boutures saines certifiées.",
                "urgency": "high",
                "prevention": "Boutures certifiées, contrôle pucerons, éliminer plants malades"
            },
            "animal_fievre": {
                "name": "Fièvre Animale (suspicion)",
                "confidence_keywords": ["bétail", "fièvre", "faible", "animal"],
                "symptoms": ["Température élevée", "Perte appétit", "Faiblesse"],
                "treatment": "CONSULTER vétérinaire RAPIDEMENT. Isoler animal. Eau fraîche disponible.",
                "urgency": "high",
                "prevention": "Vaccination, vermifugation, abri ombragé"
            }
        }
        
        # Maladies par culture pour reconnaissance rapide
        self.crop_diseases = {
            "maïs": ["mais_taches_jaunes", "mais_rouille"],
            "tomate": ["tomate_mildiou"],
            "sorgho": ["sorgho_charbon"],
            "manioc": ["manioc_mosaique"],
            "bétail": ["animal_fievre"]
        }
    
    def analyze_image_simple(self, image_data: bytes, text_description: str = "") -> dict:
        """
        Analyse simple basée sur le texte ET détection basique image
        (En attendant le vrai modèle TensorFlow)
        """
        text_lower = text_description.lower()
        
        # Déterminer la culture mentionnée
        detected_crop = None
        for crop in self.crop_diseases.keys():
            if crop in text_lower:
                detected_crop = crop
                break
        
        # Analyser l'image (version simple - détection couleur dominante)
        try:
            img = Image.open(BytesIO(image_data))
            img_array = np.array(img.resize((100, 100)))
            
            # Calculer couleur dominante
            avg_color = img_array.mean(axis=(0, 1))
            r, g, b = avg_color[:3] if len(avg_color) >= 3 else (0, 0, 0)
            
            # Détection basique selon couleur
            color_hints = []
            if r > g and r > b and r > 150:
                color_hints.extend(["rouille", "orange", "rouge"])
            if g > r and g > b and g > 100:
                color_hints.extend(["vert", "sain"])
            if r > 150 and g > 150:
                color_hints.extend(["jaune", "chlorose"])
            if r < 100 and g < 100 and b < 100:
                color_hints.extend(["noir", "pourri", "mort"])
        except Exception as e:
            print(f"Erreur analyse image: {e}")
            color_hints = []
        
        # Combiner texte + indices couleur pour matching
        all_keywords = text_lower.split() + color_hints
        
        # Trouver la maladie la plus probable
        best_match = None
        best_score = 0
        
        for disease_id, disease_info in self.diseases_database.items():
            # Score basé sur correspondance mots-clés
            score = sum(1 for kw in disease_info["confidence_keywords"] 
                       if any(kw in word for word in all_keywords))
            
            # Bonus si culture correspond
            if detected_crop and disease_id.startswith(detected_crop.replace("é", "e")):
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = disease_id
        
        # Si pas de match, réponse générique
        if best_match is None or best_score < 2:
            return {
                "disease_detected": "Indéterminé",
                "confidence": 0.3,
                "analysis": "L'analyse nécessite plus d'informations. Un expert va examiner votre photo.",
                "recommendations": "Prenez plusieurs photos (plante entière, feuilles, tiges). Décrivez les symptômes en détail.",
                "urgency": "medium",
                "requires_expert": True
            }
        
        disease = self.diseases_database[best_match]
        confidence = min(0.5 + (best_score * 0.1), 0.95)
        
        return {
            "disease_detected": disease["name"],
            "confidence": round(confidence, 2),
            "symptoms": disease["symptoms"],
            "treatment": disease["treatment"],
            "prevention": disease["prevention"],
            "urgency": disease["urgency"],
            "analysis": f"Détection probable de {disease['name']} (confiance: {int(confidence*100)}%). " + 
                       f"Symptômes typiques: {', '.join(disease['symptoms'])}.",
            "recommendations": disease["treatment"],
            "requires_expert": confidence < 0.7
        }
    
    def analyze_with_tensorflow(self, image_data: bytes) -> dict:
        """
        Analyse avec TensorFlow Lite (à implémenter)
        Pour l'instant, retourne vers analyse simple
        """
        # TODO: Charger modèle TensorFlow Lite
        # model = tf.lite.Interpreter(model_path="models/plant_disease.tflite")
        # predictions = model.predict(preprocessed_image)
        
        return self.analyze_image_simple(image_data, "")

cv_engine = LocalComputerVision()

# ==========================================
# MODULE IA TEXTE (NLP Local - amélioré)
# ==========================================

class AITriageEngine:
    def __init__(self):
        self.urgency_keywords = {
            "high": ["urgence", "urgent", "grave", "danger", "sang", "brûlure", "piraté", 
                    "volé", "mort", "mourir", "pourrir", "invasion", "attaque"],
            "medium": ["problème", "aide", "rapidement", "besoin", "important", "malade"],
            "low": ["conseil", "information", "question", "quand", "comment", "préventif"]
        }
        
        self.category_keywords = {
            "agriculture": ["maïs", "sorgho", "mil", "culture", "plante", "champ", "récolte", 
                          "bétail", "irrigation", "tomate", "oignon", "arachide", "coton",
                          "manioc", "riz", "feuille", "insecte", "parasite", "engrais"],
            "health": ["fièvre", "malade", "douleur", "enfant", "bébé", "santé", "médecin", 
                      "médicament", "blessure", "toux", "paludisme", "diarrhée"],
            "cybersecurity": ["arnaque", "pirate", "mobile money", "code", "mot de passe", 
                            "orange money", "sms suspect", "compte", "fraude", "virus"]
        }
    
    def classify(self, text: str):
        text_lower = text.lower()
        
        # Catégorie
        category_scores = {}
        for cat, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            category_scores[cat] = score
        
        category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else "agriculture"
        confidence = category_scores[category] / (len(self.category_keywords[category]) + 1)
        
        # Urgence
        urgency = "low"
        for level, keywords in self.urgency_keywords.items():
            if any(kw in text_lower for kw in keywords):
                urgency = level
                break
        
        keywords = [word for word in text_lower.split() if len(word) > 3][:5]
        
        return {
            "category": category,
            "urgency": urgency,
            "confidence": float(confidence),
            "keywords": keywords
        }

ai_engine = AITriageEngine()

# ==========================================
# ROUTES API
# ==========================================

@app.post("/api/auth/login")
async def login(data: ExpertLogin, db: Session = Depends(get_db)):
    expert = db.query(Expert).filter(Expert.email == data.email).first()
    if not expert or not verify_password(data.password, expert.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = f"token_{expert.id}_{datetime.utcnow().timestamp()}"
    return {"token": token, "expert": {"id": expert.id, "name": expert.full_name}}

@app.post("/api/webhooks/incoming-sms")
async def incoming_sms(data: MessageCreate, db: Session = Depends(get_db)):
    # 1. Utilisateur
    user = db.query(User).filter(User.phone_number == data.phone_number).first()
    if not user:
        user = User(phone_number=data.phone_number)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # 2. Analyse IA texte
    ai_result = ai_engine.classify(data.content)
    
    # 3. Analyse photo si présente
    photo_analysis = None
    photo_path = None
    
    if data.photo_base64:
        try:
            # Décoder base64
            photo_string = data.photo_base64
            if ',' in photo_string:
                photo_string = photo_string.split(',')[1]
            
            photo_data = base64.b64decode(photo_string)
            
            # Analyser avec IA locale
            photo_analysis_result = cv_engine.analyze_image_simple(photo_data, data.content)
            photo_analysis = json.dumps(photo_analysis_result)
            
            # Sauvegarder photo (optionnel)
            os.makedirs("uploads", exist_ok=True)
            photo_path = f"uploads/{user.id}_{datetime.utcnow().timestamp()}.jpg"
            with open(photo_path, "wb") as f:
                f.write(photo_data)
            
            # Ajuster urgence si maladie grave détectée
            if photo_analysis_result.get("urgency") == "high":
                ai_result["urgency"] = "high"
                
        except Exception as e:
            print(f"Erreur analyse photo: {e}")
            photo_analysis = json.dumps({"error": str(e)})
    
    # 4. Créer ticket
    ticket = Ticket(
        user_id=user.id,
        category=ai_result["category"],
        urgency=ai_result["urgency"],
        ai_confidence_score=ai_result["confidence"],
        ai_extracted_keywords=json.dumps(ai_result["keywords"]),
        ai_photo_analysis=photo_analysis,
        photo_path=photo_path,
        status="open"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    
    # 5. Message
    message = Message(
        ticket_id=ticket.id,
        sender_type="user",
        sender_id=user.id,
        content=data.content,
        channel=data.channel
    )
    db.add(message)
    db.commit()
    
    # 6. Retourner résultat avec analyse photo
    response = {
        "status": "success",
        "ticket_id": ticket.id,
        "ai_analysis": ai_result
    }
    
    if photo_analysis:
        response["photo_analysis"] = json.loads(photo_analysis)
    
    return response

@app.get("/api/user-tickets")
async def get_user_tickets(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        return []
    
    tickets = db.query(Ticket).filter(Ticket.user_id == user.id).order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        last_msg = db.query(Message).filter(Message.ticket_id == ticket.id).order_by(Message.sent_at.desc()).first()
        result.append({
            "id": ticket.id,
            "category": ticket.category,
            "urgency": ticket.urgency,
            "status": ticket.status,
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else None,
            "has_photo": ticket.photo_path is not None
        })
    
    return result

@app.get("/api/tickets")
async def get_tickets(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Ticket)
    if status:
        query = query.filter(Ticket.status == status)
    
    tickets = query.order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        user = db.query(User).filter(User.id == ticket.user_id).first()
        last_msg = db.query(Message).filter(Message.ticket_id == ticket.id).order_by(Message.sent_at.desc()).first()
        
        result.append({
            "id": ticket.id,
            "user_phone": user.phone_number if user and not user.is_anonymized else "Anonymisé",
            "category": ticket.category,
            "urgency": ticket.urgency,
            "status": ticket.status,
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else None,
            "ai_confidence": ticket.ai_confidence_score,
            "has_photo": ticket.photo_path is not None,
            "has_photo_analysis": ticket.ai_photo_analysis is not None
        })
    
    return result

@app.get("/api/tickets/{ticket_id}")
async def get_ticket_detail(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    user = db.query(User).filter(User.id == ticket.user_id).first()
    messages = db.query(Message).filter(Message.ticket_id == ticket_id).order_by(Message.sent_at).all()
    
    keywords = json.loads(ticket.ai_extracted_keywords) if ticket.ai_extracted_keywords else []
    photo_analysis = json.loads(ticket.ai_photo_analysis) if ticket.ai_photo_analysis else None
    
    return {
        "ticket": {
            "id": ticket.id,
            "category": ticket.category,
            "urgency": ticket.urgency,
            "status": ticket.status,
            "keywords": keywords,
            "confidence": ticket.ai_confidence_score,
            "photo_path": ticket.photo_path,
            "photo_analysis": photo_analysis,
            "created_at": ticket.created_at
        },
        "user": {
            "phone": user.phone_number if not user.is_anonymized else "Anonymisé",
            "name": user.name,
            "location": user.location
        },
        "messages": [{
            "content": msg.content,
            "sender_type": msg.sender_type,
            "sent_at": msg.sent_at
        } for msg in messages]
    }

@app.post("/api/tickets/{ticket_id}/reply")
async def reply_to_ticket(ticket_id: int, content: dict, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    message = Message(
        ticket_id=ticket_id,
        sender_type="expert",
        sender_id=1,
        content=content["message"],
        channel="web"
    )
    db.add(message)
    
    if not ticket.expert_id:
        ticket.expert_id = 1
        ticket.status = "assigned"
    
    db.commit()
    return {"status": "success"}

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_tickets = db.query(Ticket).count()
    open_tickets = db.query(Ticket).filter(Ticket.status == "open").count()
    resolved_today = db.query(Ticket).filter(
        Ticket.status == "resolved",
        Ticket.resolved_at >= datetime.utcnow().date()
    ).count()
    tickets_with_photos = db.query(Ticket).filter(Ticket.photo_path.isnot(None)).count()
    
    return {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "resolved_today": resolved_today,
        "tickets_with_photos": tickets_with_photos
    }

@app.get("/")
async def root():
    return {
        "message": "RESOLVE HUB API v2.0 - IA Locale Opérationnelle",
        "features": ["Computer Vision Local", "NLP Local", "Offline Ready"]
    }

@app.post("/api/create-test-expert")
async def create_test_expert(db: Session = Depends(get_db)):
    existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
    if existing:
        return {"message": "Expert already exists", "email": "test@resolvehub.bf", "password": "test123"}
    
    expert = Expert(
        email="test@resolvehub.bf",
        password_hash=hash_password("test123"),
        full_name="Expert Test CV",
        specialization="agriculture",
        is_active=True
    )
    db.add(expert)
    db.commit()
    
    return {"message": "Expert created", "email": "test@resolvehub.bf", "password": "test123"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)