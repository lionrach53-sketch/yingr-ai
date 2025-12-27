from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import uuid
import json
import os
import time
import hashlib
import shutil
from enum import Enum
import logging

# Configuration logging AVANT d'importer mongodb
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MongoDB (avec gestion d'erreur)
try:
    from mongodb import db
    from pymongo import MongoClient, DESCENDING
    logger.info("✅ Module MongoDB importé")
except Exception as e:
    logger.warning(f"⚠️  Impossible d'importer MongoDB: {e}")
    db = None
    MongoClient = None
    DESCENDING = None

# Réduire les logs MongoDB
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
logging.getLogger('pymongo.connection').setLevel(logging.WARNING)
logging.getLogger('pymongo.command').setLevel(logging.WARNING)
logging.getLogger('pymongo.serverSelection').setLevel(logging.WARNING)

# AI routes yingre ai
try:
    from ai.routes import ai_chat, ai_ingest
    from ai.service.rag import rag
except ImportError:
    try:
        from backend.ai.routes import ai_chat, ai_ingest
        from backend.ai.service.rag import rag
    except ImportError:
        ai_chat = None
        ai_ingest = None
        rag = None  
# Log import status for debugging OpenAPI visibility (safe if logger not yet configured)
try:
    _logger = logger
except NameError:
    # logger not defined yet (logging configured later) — fall back to print
    print(f"AI routes import status: ai_chat={ai_chat is not None}, ai_ingest={ai_ingest is not None}")
else:
    if ai_chat is None or ai_ingest is None:
        _logger.warning(f"AI routes import status: ai_chat={ai_chat is not None}, ai_ingest={ai_ingest is not None}")
    else:
        _logger.info("AI routes imported successfully: ai_chat and ai_ingest available")


# Initialisation
app = FastAPI(
    title="IA Souveraine Burkina - API Expert",
    version="2.0.0",
    description="API pour le panel d'expertise de l'IA souveraine du Burkina Faso",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register AI routers after `app` is created using a robust dynamic loader
def load_and_register_ai_routes(app):
    """Try multiple import strategies and ensure project root is on sys.path,
    then register available AI routers to the FastAPI app."""
    import sys
    import importlib

    logger.info("=" * 60)
    logger.info("STARTING AI ROUTES LOADER")
    logger.info("=" * 60)

    backend_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(backend_dir)
    logger.info(f"Backend dir: {backend_dir}")
    logger.info(f"Project root: {project_root}")

    # Ensure project root is on sys.path so absolute imports like "ai.routes"
    # or "backend.ai.routes" can be resolved regardless of working directory.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    candidates = ["ai.routes", "backend.ai.routes"]
    registered = []

    for cand in candidates:
        try:
            logger.debug(f"Attempting import candidate: {cand}")
            mod = importlib.import_module(cand)
            ai_chat_mod = getattr(mod, "ai_chat", None)
            ai_ingest_mod = getattr(mod, "ai_ingest", None)

            if ai_chat_mod and hasattr(ai_chat_mod, "router"):
                app.include_router(ai_chat_mod.router)
                registered.append("ai_chat")

            if ai_ingest_mod and hasattr(ai_ingest_mod, "router"):
                app.include_router(ai_ingest_mod.router)
                registered.append("ai_ingest")

            if registered:
                logger.info(f"Registered AI routers: {registered} from {cand}")
                return

        except Exception as e:
            logger.debug(f"Could not import {cand}: {e}", exc_info=True)

    # Fallback: try to load modules directly from files under backend/ai/routes
    if not registered:
        routes_dir = os.path.join(backend_dir, "ai", "routes")
        try:
            import importlib.util
            import types
            # Ensure package modules exist so relative imports inside the files work
            backend_pkg = "backend"
            backend_ai_pkg = "backend.ai"
            backend_ai_routes_pkg = "backend.ai.routes"

            if backend_pkg not in sys.modules:
                mod = types.ModuleType(backend_pkg)
                mod.__path__ = [backend_dir]
                sys.modules[backend_pkg] = mod

            ai_dir = os.path.join(backend_dir, "ai")
            if backend_ai_pkg not in sys.modules:
                mod = types.ModuleType(backend_ai_pkg)
                mod.__path__ = [ai_dir]
                sys.modules[backend_ai_pkg] = mod

            if backend_ai_routes_pkg not in sys.modules:
                mod = types.ModuleType(backend_ai_routes_pkg)
                mod.__path__ = [routes_dir]
                sys.modules[backend_ai_routes_pkg] = mod

            for fname in ("ai_chat.py", "ai_ingest.py"):
                fpath = os.path.join(routes_dir, fname)
                logger.debug(f"Checking file path: {fpath}")
                if not os.path.exists(fpath):
                    logger.debug(f"File does not exist: {fpath}")
                    continue
                try:
                    module_name = f"{backend_ai_routes_pkg}.{os.path.splitext(fname)[0]}"
                    logger.debug(f"Preparing to load module from file {fpath} as {module_name}")
                    spec = importlib.util.spec_from_file_location(module_name, fpath)
                    mod = importlib.util.module_from_spec(spec)
                    # Set package so relative imports like '..service' work
                    mod.__package__ = backend_ai_routes_pkg
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)

                    # router may be defined at module-level
                    route_mod = mod
                    logger.debug(f"Loaded module {module_name}; attributes: {list(dir(mod))[:50]}")
                    if hasattr(route_mod, "router"):
                        app.include_router(route_mod.router)
                        registered.append(os.path.splitext(fname)[0])
                        logger.debug(f"Registered router from {module_name}")

                except Exception as e:
                    logger.debug(f"Failed to load AI route from file {fpath}: {e}")

            if registered:
                logger.info(f"Registered AI routers from files: {registered}")
                return
        except Exception as e:
            logger.debug(f"File-based AI loader failed: {e}")

    logger.warning("No AI routers registered. Check import paths and working directory.")


# Execute loader
load_and_register_ai_routes(app)

# Forcer l'import et l'inclusion des routes IA même si le loader dynamique échoue
try:
    from ai.routes import ai_chat, ai_ingest
    print("[DEBUG] ai_chat et ai_ingest importés depuis ai.routes")
    if hasattr(ai_chat, "router"):
        app.include_router(ai_chat.router)
        print("[DEBUG] Router ai_chat inclus")
    if hasattr(ai_ingest, "router"):
        app.include_router(ai_ingest.router)
        print("[DEBUG] Router ai_ingest inclus")
except Exception as e:
    print(f"[DEBUG] Erreur import/include routers: {e}")

# CORS configuration - autoriser front local + front déployé sur la VM
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5175",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:5176",
        # Frontend user prod servi par `serve` sur la VM
        "http://34.173.253.235:4173",
        # Frontend admin prod (autre port sur la même VM)
        "http://34.173.253.235:4174",
        # Frontend expert prod (encore un autre port)
        "http://34.173.253.235:4175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité
security = HTTPBearer()

# Configuration
EXPERT_KEY = "expert-burkina-2024"  # Token d'accès expert
ADMIN_KEY = "admin-souverain-burkina-2024"  # Token admin séparé
DB_FILE = "data/expert_db.json"
LOGS_FILE = "data/expert_logs.json"
UPLOAD_DIR = "uploads"
START_TIME = time.time()

# Créer les répertoires nécessaires
os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Modèles Pydantic
class ExpertLogin(BaseModel):
    username: str = Field(..., json_schema_extra={"example": "expert1"})
    password: str = Field(..., json_schema_extra={"example": "expert123"})
    
class ExpertInfo(BaseModel):
    id: str = Field(..., example="exp_001")
    name: str = Field(..., example="Dr. Ibrahim Traoré")
    email: EmailStr = Field(..., example="expert@ia-burkina.bf")
    level: int = Field(..., example=3)
    specialty: str = Field(..., example="Agriculture")
    contributions_count: int = Field(..., example=42)
    validation_score: float = Field(..., example=94.5)
    join_date: str = Field(..., example="2024-01-01")

class KnowledgeSubmission(BaseModel):
    title: str = Field(..., min_length=5, max_length=200, example="Les techniques agricoles traditionnelles")
    content: str = Field(..., min_length=20, example="Les techniques agricoles traditionnelles au Burkina Faso incluent...")
    category: str = Field(..., example="Agriculture")
    source: Optional[str] = Field(None, example="Ministère de l'Agriculture")
    tags: List[str] = Field(default=[], example=["agriculture", "tradition", "burkina"])

class DocumentUpload(BaseModel):
    category: str = Field(..., example="Agriculture")
    description: Optional[str] = Field(None, example="Document sur les techniques agricoles")

class Contribution(BaseModel):
    id: str = Field(..., example="1")
    title: str = Field(..., example="Les techniques agricoles traditionnelles")
    content: str = Field(..., example="Les techniques agricoles traditionnelles au Burkina Faso incluent...")
    category: str = Field(..., example="Agriculture")
    status: str = Field(..., example="validated")
    createdAt: str = Field(..., example="2024-01-15T10:30:00")
    validatedAt: Optional[str] = Field(None, example="2024-01-16T14:20:00")
    expertName: str = Field(..., example="Dr. Ibrahim Traoré")
    expertId: str = Field(..., example="exp_001")

class ValidationItem(BaseModel):
    id: str = Field(..., example="4")
    title: str = Field(..., example="Recette de soupe de niébé")
    content: str = Field(..., example="La soupe de niébé est un plat traditionnel burkinabè riche en protéines...")
    category: str = Field(..., example="Cuisine")
    submittedBy: str = Field(..., example="Chef Aminata")
    submittedAt: str = Field(..., example="2024-01-13T16:30:00")
    expertId: str = Field(..., example="exp_004")

class ValidationRequest(BaseModel):
    isValid: bool = Field(..., example=True)
    corrections: Optional[str] = Field(None, example="Quelques corrections mineures à apporter...")

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2024-01-15T10:30:00")
    uptime: str = Field(..., example="2 jours 4 heures")
    version: str = Field(..., example="2.0.0")
    expert_count: int = Field(..., example=5)
    pending_validations: int = Field(..., example=3)

class UploadResponse(BaseModel):
    message: str = Field(..., example="Document uploadé avec succès")
    filename: str = Field(..., example="document.pdf")
    size: str = Field(..., example="2.4 MB")
    uploaded_at: str = Field(..., example="2024-01-15T10:30:00")
    file_url: str = Field(..., example="/uploads/document.pdf")

class ChatMessage(BaseModel):
    message: str = Field(..., example="Bonjour, comment ça va ?")
    category: str = Field(..., example="general")
    timestamp: Optional[str] = Field(None, example="2024-01-15T10:30:00")

class ChatResponse(BaseModel):
    response: str = Field(..., example="Je vais bien, merci !")
    confidence: float = Field(..., example=0.85)
    sources: List[str] = Field([], example=["Base de connaissances agriculture", "Documentation santé"])
    conversation_id: str = Field(..., example="conv_12345")

class Category(BaseModel):
    id: str = Field(..., example="general")
    name: str = Field(..., example="Général")
    description: Optional[str] = Field(None, example="Questions générales")

# Modèle de requête
class GuestChatRequest(BaseModel):
    message: str
    category: str = "general"
    session_id: Optional[str] = None  # UUID pour la session

# Réponse
class GuestChatResponse(BaseModel):
    conversation_id: str
    response: str
    timestamp: str



# Base de données des experts (en production, utiliser une vraie base de données)
EXPERT_DATABASE = {
    "expert1": {
        "id": "exp_001",
        "name": "Dr. Ibrahim Traoré",
        "email": "expert@ia-burkina.bf",
        "password": "expert123",  # En production, utiliser bcrypt
        "level": 3,
        "specialty": "Agriculture",
        "contributions_count": 42,
        "validation_score": 94.5,
        "join_date": "2024-01-01"
    },
    "expert2": {
        "id": "exp_002",
        "name": "Dr. Aïcha Diallo",
        "email": "aicha.diallo@example.com",
        "password": "expert456",
        "level": 2,
        "specialty": "Santé",
        "contributions_count": 28,
        "validation_score": 88.2,
        "join_date": "2024-01-05"
    },
    "expert3": {
        "id": "exp_003",
        "name": "Pr. Moussa Sawadogo",
        "email": "m.sawadogo@example.com",
        "password": "expert789",
        "level": 4,
        "specialty": "Histoire",
        "contributions_count": 67,
        "validation_score": 96.8,
        "join_date": "2023-12-15"
    }
}

# Services utilitaires
class Database:
    @staticmethod
    def init():
        """Initialise la base de données expert"""
        if not os.path.exists(DB_FILE):
            default_data = {
                "contributions": [
                    {
                        "id": "1",
                        "title": "Les techniques agricoles traditionnelles",
                        "content": "Les techniques agricoles traditionnelles au Burkina Faso incluent la rotation des cultures, l'utilisation de compost naturel et les techniques d'irrigation traditionnelles comme les diguettes.",
                        "category": "Agriculture",
                        "source": "Ministère de l'Agriculture",
                        "tags": ["agriculture", "tradition", "techniques"],
                        "status": "validated",
                        "expertId": "exp_001",
                        "expertName": "Dr. Ibrahim Traoré",
                        "createdAt": "2024-01-15T10:30:00",
                        "validatedAt": "2024-01-16T14:20:00"
                    },
                    {
                        "id": "2",
                        "title": "Plantes médicinales locales",
                        "content": "Le moringa est une plante aux multiples vertus : riche en vitamines et minéraux, elle est utilisée pour lutter contre la malnutrition. Le neem est quant à lui utilisé pour ses propriétés antiseptiques.",
                        "category": "Santé",
                        "source": "Institut de recherche en santé",
                        "tags": ["santé", "plantes", "médecine traditionnelle"],
                        "status": "pending",
                        "expertId": "exp_002",
                        "expertName": "Dr. Aïcha Diallo",
                        "createdAt": "2024-01-14T15:45:00",
                        "validatedAt": None
                    }
                ],
                "validation_queue": [
                    {
                        "id": "4",
                        "title": "Recette de soupe de niébé",
                        "content": "La soupe de niébé est un plat traditionnel burkinabè riche en protéines. Ingrédients : niébé, oignons, tomates, huile d'arachide, piment. Cuire le niébé pendant 2 heures, puis ajouter les légumes.",
                        "category": "Cuisine",
                        "submittedBy": "Chef Aminata",
                        "submittedAt": "2024-01-13T16:30:00",
                        "expertId": "exp_004"
                    },
                    {
                        "id": "5",
                        "title": "Techniques de conservation des céréales",
                        "content": "Les techniques traditionnelles de conservation incluent le séchage au soleil, le stockage dans des greniers surélevés et l'utilisation de cendres ou de plantes répulsives contre les insectes.",
                        "category": "Agriculture",
                        "submittedBy": "Ing. Boubacar",
                        "submittedAt": "2024-01-12T11:15:00",
                        "expertId": "exp_005"
                    }
                ],
                "documents": [
                    {
                        "id": "doc_001",
                        "filename": "guide_agriculture.pdf",
                        "category": "Agriculture",
                        "size": "2.4 MB",
                        "uploaded_at": "2024-01-10T09:15:00",
                        "uploaded_by": "exp_001",
                        "description": "Guide des techniques agricoles traditionnelles"
                    }
                ],
                "stats": {
                    "total_contributions": 156,
                    "pending_validations": 12,
                    "total_experts": 5,
                    "validation_rate": 92.5,
                    "documents_count": 8
                }
            }
            Database.save(default_data)
            logger.info("Base de données expert initialisée")
    
    @staticmethod
    def load():
        """[DÉPRÉCIÉ] Ancienne lecture JSON remplacée par MongoDB.

        Conservée uniquement pour compatibilité, mais ne doit plus être utilisée.
        """
        logger.warning("Database.load() est déprécié - utiliser MongoDB à la place")
        return {"contributions": [], "validation_queue": [], "documents": [], "stats": {}, "api_keys": []}
    
    @staticmethod
    def save(data):
        """[DÉPRÉCIÉ] Ancienne écriture JSON remplacée par MongoDB.

        Ne fait plus rien, gardée pour compatibilité.
        """
        logger.warning("Database.save() est déprécié - aucune écriture effectuée")
        return
    
    @staticmethod
    def add_log(action: str, expert_id: str = "anonymous", details: dict = None):
        """[DÉPRÉCIÉ] Redirigé vers les logs admin MongoDB pour compatibilité."""
        try:
            if db is not None:
                # Utiliser la collection admin_logs de MongoDB
                db.add_admin_log(action, admin_id=expert_id, details=details or {})
            else:
                logger.warning(f"add_log sans MongoDB: {action} ({expert_id})")
        except Exception as e:
            logger.error(f"Erreur log (MongoDB): {e}")

# Dépendances
async def verify_expert(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie l'authentification expert"""
    if credentials.credentials != EXPERT_KEY:
        if db is not None:
            db.add_admin_log("auth_failed", admin_id="expert_unknown", details={"reason": "invalid_token"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expert invalide",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie l'authentification admin"""
    if credentials.credentials != ADMIN_KEY:
        if db is not None:
            db.add_admin_log("auth_failed_admin", admin_id="unknown", details={"reason": "invalid_admin_key"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé d'administration invalide",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

# Routes publiques
@app.get("/", tags=["Public"])
async def root():
    """Route racine"""
    return {
        "application": "IA Souveraine Burkina - Panel Expert",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "expert_login": "/api/expert/login",
            "expert_docs": "/docs"
        }
    }


# Importer le vrai moteur IA (RAGService)
try:
    from ai.service.rag import RAGService
except ImportError:
    from backend.ai.service.rag import RAGService

rag = RAGService()

@app.post("/api/chat/guest", response_model=GuestChatResponse)
async def guest_chat(req: GuestChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        # Appel au moteur IA (RAG)
        answer, context = rag.ask(req.message, k=5)
        conversation_entry = {
            "user_id": session_id,
            "category": req.category,
            "messages": [
                {"role": "user", "content": req.message, "timestamp": datetime.utcnow()},
                {"role": "ai", "content": answer, "timestamp": datetime.utcnow()}
            ],
            "timestamp": datetime.utcnow()
        }
        conversation_id = db.save_chat_conversation(conversation_entry)
        return {
            "conversation_id": conversation_id,
            "response": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA invité: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Public"])
async def health():
    """Health check endpoint"""
    try:
        stats = db.get_system_stats()  # REMPLACE Database.load()
        
        uptime_seconds = time.time() - START_TIME
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime=f"{days} jours {hours} heures",
            version="2.0.0",
            expert_count=3,  # À adapter pour récupérer depuis db
            pending_validations=stats.get("pending_validations", 0)
        )
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            uptime="0 jours 0 heures",
            version="2.0.0",
            expert_count=0,
            pending_validations=0
        )

# Routes expert
@app.post("/api/expert/login", response_model=ExpertInfo, tags=["Expert"])
async def expert_login(auth: ExpertLogin):
    """Authentification expert"""
    try:
        expert = EXPERT_DATABASE.get(auth.username)
        
        if not expert or expert["password"] != auth.password:
            if db is not None:
                db.add_admin_log("login_failed", admin_id="expert_unknown", details={"username": auth.username})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Identifiants invalides"
            )
        
        # Ne pas renvoyer le mot de passe
        expert_info = expert.copy()
        del expert_info["password"]
        
        if db is not None:
            db.add_admin_log("login_success", admin_id=expert_info["id"], details={"name": expert_info["name"]})
        logger.info(f"Expert connecté: {expert_info['name']}")
        
        return expert_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur login: {e}")
        raise HTTPException(status_code=500, detail="Erreur d'authentification")

@app.post("/api/expert/knowledge", tags=["Expert"])
async def submit_knowledge(
    knowledge: KnowledgeSubmission,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_expert)
):
    """Soumettre une nouvelle connaissance - Version MongoDB"""
    try:
        expert = EXPERT_DATABASE["expert1"]  # À adapter
        
        contribution_data = {
            "id": str(uuid.uuid4())[:8],
            "title": knowledge.title,
            "content": knowledge.content,
            "category": knowledge.category,
            "source": knowledge.source,
            "tags": knowledge.tags,
            "status": "pending",
            "expertId": expert["id"],
            "expertName": expert["name"],
            "createdAt": datetime.now()
        }
        
        # AJOUT DANS MONGODB
        contribution_id = db.add_contribution(contribution_data)
        
        # Log dans MongoDB
        background_tasks.add_task(
            db.add_admin_log,
            "knowledge_submitted",
            expert["id"],
            {
                "knowledge_id": contribution_id,
                "title": knowledge.title,
                "category": knowledge.category
            }
        )
        
        logger.info(f"📝 Connaissance soumise dans MongoDB: {knowledge.title}")
        
        return {
            "message": "Connaissance soumise avec succès",
            "id": contribution_data["id"],
            "mongo_id": contribution_id,
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Erreur soumission connaissance: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/api/expert/upload", response_model=UploadResponse, tags=["Expert"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    _: bool = Depends(verify_expert)
):
    """Uploader un document"""
    try:
        # Vérifier le type de fichier
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.png']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non autorisé. Types acceptés: {', '.join(allowed_extensions)}"
            )
        
        # Générer un nom de fichier unique (code manquant)
        file_id = str(uuid.uuid4())[:8]
        original_name = os.path.splitext(file.filename)[0]
        safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        new_filename = f"{safe_name}_{file_id}{file_ext}"
        
        # Sauvegarder le fichier
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Obtenir la taille du fichier
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB"
        
        # PRÉPAREZ les données pour MongoDB
        document_record = {
            "id": f"doc_{file_id}",
            "filename": new_filename,
            "original_name": file.filename,
            "category": category,
            "description": description,
            "size": size_str,
            "uploaded_by": "exp_001",  # En pratique, extraire du token
            "file_path": file_path
            # uploaded_at sera ajouté automatiquement par db.add_document()
        }
        
        # AJOUT à MongoDB
        doc_id = db.add_document(document_record)
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "document_uploaded",
            "exp_001",
            {
                "filename": new_filename,
                "category": category,
                "size": size_str
            }
        )
        
        logger.info(f"Document uploadé: {new_filename}")
        
        return UploadResponse(
            message="Document uploadé avec succès",
            filename=file.filename,
            size=size_str,
            uploaded_at=datetime.now().isoformat(),
            file_url=f"/uploads/{new_filename}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

@app.get("/api/expert/contributions", response_model=List[Contribution], tags=["Expert"])
async def get_contributions(_: bool = Depends(verify_expert)):
    """Obtenir toutes les contributions depuis MongoDB"""
    try:
        # Utiliser MongoDB au lieu du fichier JSON
        contributions_cursor = db.contributions.find({}).sort("createdAt", DESCENDING)
        contributions = list(contributions_cursor)
        
        logger.info(f"Récupéré {len(contributions)} contributions depuis MongoDB")
        
        if not contributions:
            return []
        
        # Formater les données pour le modèle Pydantic
        formatted_contributions = []
        for contrib in contributions:
            # S'assurer que le champ 'id' existe
            contrib_id = contrib.get('id')
            if not contrib_id:
                # Utiliser l'ObjectId MongoDB comme fallback
                contrib_id = str(contrib.get('_id', ''))
            
            # Convertir les dates
            created_at = contrib.get('createdAt')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            
            validated_at = contrib.get('validatedAt')
            if validated_at and isinstance(validated_at, datetime):
                validated_at = validated_at.isoformat()
            
            formatted_contributions.append({
                "id": contrib_id,
                "title": contrib.get('title', 'Sans titre'),
                "content": contrib.get('content', ''),
                "category": contrib.get('category', 'general'),
                "status": contrib.get('status', 'pending'),
                "createdAt": created_at or datetime.now().isoformat(),
                "validatedAt": validated_at,
                "expertName": contrib.get('expertName', 'Expert inconnu'),
                "expertId": contrib.get('expertId', 'exp_000')
            })
        
        return formatted_contributions
        
    except Exception as e:
        logger.error(f"Erreur get_contributions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur MongoDB: {str(e)}")

@app.get("/api/expert/validation-queue", response_model=List[ValidationItem], tags=["Expert"])
async def get_validation_queue(_: bool = Depends(verify_expert)):
    """Obtenir la file de validation depuis MongoDB"""
    try:
        # Utiliser MongoDB au lieu du fichier JSON
        queue_cursor = db.validation_queue.find({"validated": False}).sort("submittedAt", DESCENDING)
        queue = list(queue_cursor)
        
        logger.info(f"Récupéré {len(queue)} éléments dans la file de validation")
        
        if not queue:
            return []
        
        # Formater les données pour le modèle Pydantic
        formatted_queue = []
        for item in queue:
            # S'assurer que le champ 'id' existe
            item_id = item.get('id')
            if not item_id:
                item_id = str(item.get('_id', ''))
            
            # Convertir les dates
            submitted_at = item.get('submittedAt')
            if isinstance(submitted_at, datetime):
                submitted_at = submitted_at.isoformat()
            
            formatted_queue.append({
                "id": item_id,
                "title": item.get('title', 'Sans titre'),
                "content": item.get('content', ''),
                "category": item.get('category', 'general'),
                "submittedBy": item.get('submittedBy', 'Anonyme'),
                "submittedAt": submitted_at or datetime.now().isoformat(),
                "expertId": item.get('expertId', 'exp_000')
            })
        
        return formatted_queue
        
    except Exception as e:
        logger.error(f"Erreur get_validation_queue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur MongoDB: {str(e)}")

# 6. MODIFIEZ la fonction validate_knowledge() (ligne ~360)
@app.post("/api/expert/validate/{knowledge_id}", tags=["Expert"])
async def validate_knowledge(
    knowledge_id: str,
    validation: ValidationRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_expert)
):
    """Valider une connaissance"""
    try:
        # VALIDATION avec MongoDB
        success = db.validate_item(
            knowledge_id,
            validation.isValid,
            validation.corrections
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Connaissance non trouvée dans la file de validation"
            )
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "knowledge_validated",
            "exp_001",  # À adapter
            {
                "knowledge_id": knowledge_id,
                "isValid": validation.isValid,
                "has_corrections": bool(validation.corrections)
            }
        )
        
        logger.info(f"Connaissance validée: {knowledge_id}")
        
        return {
            "message": "Validation enregistrée avec succès",
            "knowledge_id": knowledge_id,
            "validated": True,
            "isValid": validation.isValid
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur validation: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/expert/profile", response_model=ExpertInfo, tags=["Expert"])
async def get_profile(_: bool = Depends(verify_expert)):
    """Obtenir le profil de l'expert"""
    try:
        # Pour la démo, retourner le profil de expert1
        # En pratique, extraire l'ID expert du token JWT
        expert = EXPERT_DATABASE["expert1"]
        
        # Ne pas renvoyer le mot de passe
        expert_info = expert.copy()
        del expert_info["password"]
        
        return expert_info
        
    except Exception as e:
        logger.error(f"Erreur get_profile: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/expert/stats", tags=["Expert"])
async def get_expert_stats(_: bool = Depends(verify_expert)):
    """Obtenir les statistiques pour l'expert"""
    try:
        # Récupérer les contributions et stats depuis MongoDB
        expert_id = "exp_001"  # TODO: extraire depuis le token dans une version future

        # Contributions personnelles
        expert_contributions = db.get_contributions(filter_by={"expertId": expert_id}) if db is not None else []
        validated_count = len([c for c in expert_contributions if c.get("status") == "validated"])
        pending_count = len([c for c in expert_contributions if c.get("status") == "pending"])

        # Taux de validation basé sur la file de validation MongoDB
        validation_items = db.get_validation_queue() if db is not None else []
        validated_items = [v for v in validation_items if v.get("validated")]
        validation_rate = len(validated_items) / len(validation_items) * 100 if validation_items else 0

        # Documents uploadés par l'expert
        documents_uploaded = db.documents.count_documents({"uploaded_by": expert_id}) if db is not None else 0

        # Stats globales depuis MongoDB
        global_stats = db.get_system_stats() if db is not None else {}

        return {
            "personal": {
                "total_contributions": len(expert_contributions),
                "validated_contributions": validated_count,
                "pending_contributions": pending_count,
                "validation_rate": round(validation_rate, 1),
                "documents_uploaded": documents_uploaded
            },
            "global": global_stats
        }
        
    except Exception as e:
        logger.error(f"Erreur get_expert_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Routes admin (optionnelles)
@app.get("/api/admin/logs", tags=["Admin"])
async def get_logs(
    limit: int = 100,
    _: bool = Depends(verify_admin)
):
    """Obtenir les logs (admin seulement)"""
    try:
        # RÉCUPÉRATION depuis MongoDB
        logs = db.get_admin_logs(limit)
        
        # Convertir ObjectId et datetime
        for log in logs:
            if '_id' in log:
                log['id'] = str(log['_id'])
                del log['_id']
            if 'timestamp' in log and isinstance(log['timestamp'], datetime):
                log['timestamp'] = log['timestamp'].isoformat()
        
        return logs
        
    except Exception as e:
        logger.error(f"Erreur get_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================
# MODÈLES PYDANTIC POUR ADMIN
# ============================================

class StatsResponse(BaseModel):
    total_requests: int = Field(default=0, json_schema_extra={"example": 1250})
    active_users: int = Field(default=0, json_schema_extra={"example": 42})
    documents_count: int = Field(default=0, json_schema_extra={"example": 15})
    api_version: str = Field(default="2.0.0", json_schema_extra={"example": "2.0.0"})
    uptime: str = Field(default="0 jours 0 heures", json_schema_extra={"example": "2 jours 4 heures"})
    memory_usage: str = Field(default="45%", json_schema_extra={"example": "45%"})
    requests_today: int = Field(default=0, json_schema_extra={"example": 87})
    avg_response_time: float = Field(default=0.0, json_schema_extra={"example": 120.5})
    total_conversations: int = Field(default=0, json_schema_extra={"example": 156})
    active_api_keys: int = Field(default=0, json_schema_extra={"example": 3})
    system_status: str = Field(default="healthy", json_schema_extra={"example": "healthy"})

class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=50, json_schema_extra={"example": "Application Mobile"})
    permissions: Dict[str, bool] = Field(
        default={"read": True, "write": False, "delete": False},
        json_schema_extra={"example": {"read": True, "write": True, "delete": False}}
    )

class ApiKeyResponse(BaseModel):
    id: str = Field(..., json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"})
    name: str = Field(..., json_schema_extra={"example": "Application Mobile"})
    key: str = Field(..., json_schema_extra={"example": "sk_live_1234567890abcdef"})
    created_at: datetime = Field(..., json_schema_extra={"example": "2024-01-15T10:30:00"})
    active: bool = Field(default=True, json_schema_extra={"example": True})
    last_used: Optional[datetime] = Field(None, json_schema_extra={"example": "2024-01-15T14:30:00"})
    permissions: Dict[str, bool] = Field(
        default={"read": True, "write": False, "delete": False},
        json_schema_extra={"example": {"read": True, "write": True, "delete": False}}
    )

class SystemAction(BaseModel):
    action: str = Field(..., json_schema_extra={"example": "restart"})
    force: bool = Field(default=False, json_schema_extra={"example": False})

# ============================================
# ROUTES ADMIN
# ============================================

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["Admin"])
async def get_admin_stats(_: bool = Depends(verify_admin)):
    """Obtenir les statistiques système RÉELLES depuis MongoDB"""
    try:
        from mongodb import db
        
        # Compter les documents RÉELS dans MongoDB
        documents_count = db.documents.count_documents({})
        
        # Compter les conversations RÉELLES
        conversations_count = db.chat_conversations.count_documents({})
        
        # Compter les clés API actives RÉELLES
        active_keys = db.api_keys.count_documents({"active": True})
        
        # Compter les requêtes aujourd'hui RÉELLES
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        requests_today = db.chat_conversations.count_documents({
            "timestamp": {"$gte": today_start}
        })
        
        # Calculer le temps de réponse moyen RÉEL (si stocké dans les logs)
        avg_response_time = 0.0
        recent_logs = list(db.chat_conversations.find(
            {"response_time": {"$exists": True}},
            {"response_time": 1}
        ).limit(100))
        
        if recent_logs:
            total_time = sum(log.get("response_time", 0) for log in recent_logs)
            avg_response_time = total_time / len(recent_logs)
        
        # Uptime réel
        uptime_seconds = time.time() - START_TIME
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        
        # Mémoire (optionnel - garder fixe ou calculer avec psutil)
        memory_usage = "N/A"  # Ou utiliser psutil pour des stats réelles
        
        # AJOUTEZ un log admin dans MongoDB
        db.add_admin_log("get_stats", "admin")
        
        return StatsResponse(
            total_requests=conversations_count,  # RÉEL
            active_users=0,  # Pas de tracking utilisateurs actuellement
            documents_count=documents_count,  # RÉEL
            api_version="2.0.0",
            uptime=f"{days} jours {hours} heures",
            memory_usage=memory_usage,
            requests_today=requests_today,  # RÉEL
            avg_response_time=avg_response_time,  # RÉEL
            total_conversations=conversations_count,  # RÉEL
            active_api_keys=active_keys,  # RÉEL
            system_status="healthy"
        )
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/admin/conversations", tags=["Admin"])
async def get_admin_conversations(
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(verify_admin)
):
    """Obtenir les conversations depuis MongoDB"""
    try:
        # Récupérer depuis MongoDB
        from mongodb import db
        conversations = list(db.chat_conversations.find()
            .sort("timestamp", -1)
            .skip(offset)
            .limit(limit))
        
        # Formater pour JSON
        formatted = []
        for conv in conversations:
            # Convertir ObjectId en string
            conv_id = str(conv.get("_id", ""))
            
            # Formater la date
            timestamp = conv.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            conv_data = {
                "id": conv_id,
                "user_message": conv.get("user_message", ""),
                "ai_response": conv.get("ai_response", ""),
                "category": conv.get("category", "general"),
                "conversation_id": conv.get("conversation_id", ""),
                "timestamp": timestamp
            }
            formatted.append(conv_data)
        
        total = db.chat_conversations.count_documents({})
        
        return {
            "conversations": formatted,
            "total": total,
            "page": offset // limit + 1 if limit > 0 else 1,
            "pages": (total + limit - 1) // limit if limit > 0 else 1
        }
        
    except Exception as e:
        logger.error(f"Erreur get_conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/api-keys", response_model=List[ApiKeyResponse], tags=["Admin"])
async def get_admin_api_keys(_: bool = Depends(verify_admin)):
    """Lister toutes les clés API"""
    try:
        # Récupérer directement depuis MongoDB
        api_keys_cursor = db.api_keys.find({}) if db is not None else []
        api_keys: List[Dict[str, Any]] = []

        for key in api_keys_cursor:
            item = {
                "id": key.get("id", str(key.get("_id", ""))),
                "name": key.get("name", ""),
                "key": key.get("key", ""),
                "created_at": key.get("created_at", datetime.now()),
                "active": key.get("active", True),
                "last_used": key.get("last_used"),
                "permissions": key.get("permissions", {"read": True, "write": False, "delete": False})
            }
            api_keys.append(item)

        if db is not None:
            db.add_admin_log("get_api_keys", "admin", {"count": len(api_keys)})

        return api_keys
    except Exception as e:
        logger.error(f"Erreur get_api_keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/api-keys", response_model=ApiKeyResponse, tags=["Admin"])
async def create_admin_api_key(
    key_data: ApiKeyCreate,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Créer une nouvelle clé API"""
    try:
        # Générer une clé unique
        new_key_id = str(uuid.uuid4())
        new_key_value = f"sk_live_{uuid.uuid4().hex[:24]}"
        
        new_key = {
            "id": new_key_id,
            "name": key_data.name,
            "key": new_key_value,
            "created_at": datetime.now(),
            "active": True,
            "permissions": key_data.permissions
        }
        # Sauvegarde dans MongoDB
        if db is not None:
            db.api_keys.insert_one(new_key)
        
        # Ajouter un log
        background_tasks.add_task(
            db.add_admin_log,
            "api_key_created",
            "admin",
            {"key_id": new_key_id, "name": key_data.name}
        )
        
        logger.info(f"Clé API créée: {key_data.name}")
        return new_key
        
    except Exception as e:
        logger.error(f"Erreur création clé: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/api-keys/{key_id}", tags=["Admin"])
async def revoke_admin_api_key(
    key_id: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Révoquer une clé API"""
    try:
        # Mettre à jour la clé dans MongoDB
        result = db.api_keys.update_one({"id": key_id}, {"$set": {"active": False}}) if db is not None else None

        if result is None or result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Clé API non trouvée")
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "api_key_revoked",
            "admin",
            {"key_id": key_id}
        )
        
        logger.info(f"Clé API révoquée: {key_id}")
        return {"message": "Clé API révoquée avec succès", "key_id": key_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur révocation clé: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/knowledge", tags=["Admin"])
async def get_admin_knowledge(_: bool = Depends(verify_admin)):
    """Obtenir la base de connaissances depuis MongoDB"""
    try:
        all_knowledge = []
        
        # 1. Récupérer les contributions depuis MongoDB
        contributions = db.get_all_contributions()
        for contribution in contributions:
            all_knowledge.append({
                "id": str(contribution.get("_id", contribution.get("id", "unknown"))),
                "title": contribution.get("title", "Sans titre"),
                "content": contribution.get("content", ""),
                "category": contribution.get("category", "Non catégorisé"),
                "type": "Contribution Expert",
                "status": contribution.get("status", "pending"),
                "created_at": contribution.get("createdAt", datetime.now()).isoformat() if isinstance(contribution.get("createdAt"), datetime) else contribution.get("createdAt", datetime.now().isoformat()),
                "author": contribution.get("expertName", "Inconnu"),
                "expertName": contribution.get("expertName", "Inconnu"),
                "source": "Expert MongoDB"
            })
        
        # 2. Récupérer les documents depuis MongoDB
        documents = db.get_all_documents()
        for doc in documents:
            all_knowledge.append({
                "id": str(doc.get("_id", doc.get("id", "unknown"))),
                "title": doc.get("filename", "Sans nom"),
                "content": f"Document: {doc.get('description', 'Aucune description')}",
                "category": doc.get("category", "Document"),
                "type": "Document",
                "status": doc.get("status", "uploaded"),
                "created_at": doc.get("uploaded_at", datetime.now()).isoformat() if isinstance(doc.get("uploaded_at"), datetime) else doc.get("uploaded_at", datetime.now().isoformat()),
                "author": doc.get("uploaded_by", "Inconnu"),
                "source": "Document MongoDB",
                "size": doc.get("size", "Inconnu")
            })
        
        # Log dans MongoDB
        db.add_admin_log("get_knowledge", "admin", {"count": len(all_knowledge)})
        
        logger.info(f"📊 Récupération de {len(all_knowledge)} connaissances depuis MongoDB")
        
        return all_knowledge
        
    except Exception as e:
        logger.error(f"Erreur get_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/knowledge/{item_id}", tags=["Admin"])
async def delete_admin_knowledge(
    item_id: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Supprimer un élément de la base de connaissances"""
    try:
        deleted = False
        source = ""
        
        # Chercher et supprimer dans les contributions MongoDB
        result = db.contributions.delete_one({"id": item_id})
        if result.deleted_count > 0:
            deleted = True
            source = "contributions (MongoDB)"
        
        # Chercher dans la file de validation (format: val_xxx)
        if not deleted and item_id.startswith("val_"):
            original_id = item_id[4:]  # Enlever le préfixe val_
            result = db.validation_queue.delete_one({"id": original_id})
            if result.deleted_count > 0:
                deleted = True
                source = "validation_queue (MongoDB)"
        
        # Chercher dans les documents MongoDB
        if not deleted:
            result = db.documents.delete_one({"id": item_id})
            if result.deleted_count > 0:
                deleted = True
                source = "documents (MongoDB)"
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Élément non trouvé dans MongoDB")
        
        # Log dans MongoDB
        background_tasks.add_task(
            db.add_admin_log,
            "knowledge_deleted",
            "admin",
            {"item_id": item_id, "source": source}
        )
        
        logger.info(f"Connaissance supprimée de MongoDB: {item_id} (source: {source})")
        
        return {
            "message": f"Élément supprimé de {source}",
            "item_id": item_id,
            "deleted": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur suppression connaissance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ROUTES ADMIN - GESTION RAG/FAISS
# ============================================

@app.get("/api/admin/rag/stats", tags=["Admin", "RAG"])
async def get_rag_stats(_: bool = Depends(verify_admin)):
    """Obtenir les statistiques de la base RAG/FAISS"""
    try:
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        stats = rag.vector_store.get_stats()
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erreur stats RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/rag/documents", tags=["Admin", "RAG"])
async def get_rag_documents(
    limit: int = 100,
    offset: int = 0,
    language: Optional[str] = None,
    _: bool = Depends(verify_admin)
):
    """Lister tous les documents dans la base RAG/FAISS"""
    try:
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        all_docs = rag.vector_store.get_all_metadata()
        
        # Filtrer par langue si spécifié
        if language:
            filtered = []
            for idx, meta in all_docs:
                source = meta.get('source', '')
                if f'-{language}' in source:
                    filtered.append({
                        'index': idx,
                        'text': meta.get('text', '')[:200] + '...',  # Extrait
                        'source': source,
                        'language': language,
                        'full_text_length': len(meta.get('text', ''))
                    })
            all_docs = filtered
        else:
            all_docs = [{
                'index': idx,
                'text': meta.get('text', '')[:200] + '...',
                'source': meta.get('source', ''),
                'language': _extract_language(meta.get('source', '')),
                'full_text_length': len(meta.get('text', ''))
            } for idx, meta in all_docs]
        
        # Pagination
        total = len(all_docs)
        paginated = all_docs[offset:offset+limit]
        
        return {
            "success": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "documents": paginated
        }
    
    except Exception as e:
        logger.error(f"Erreur liste documents RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_language(source: str) -> str:
    """Extrait la langue depuis la source (ex: admin-json-Histoire-fr → fr)"""
    if '-fr' in source:
        return 'fr'
    elif '-mo' in source:
        return 'mo'
    elif '-di' in source:
        return 'di'
    return 'unknown'


@app.delete("/api/admin/rag/documents/{index}", tags=["Admin", "RAG"])
async def delete_rag_document(
    index: int,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Supprimer un document spécifique du RAG par son index"""
    try:
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        # Vérifier que l'index existe
        all_docs = rag.vector_store.get_all_metadata()
        if index >= len(all_docs):
            raise HTTPException(status_code=404, detail=f"Document index {index} non trouvé")
        
        # Supprimer
        rag.vector_store.delete_by_indices([index])
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "rag_document_deleted",
            "admin",
            {"index": index}
        )
        
        return {
            "success": True,
            "message": f"Document {index} supprimé du RAG",
            "deleted_index": index
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur suppression document RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/rag/language/{language}", tags=["Admin", "RAG"])
async def delete_rag_by_language(
    language: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Supprimer tous les documents d'une langue spécifique (fr, mo, di)"""
    try:
        if language not in ['fr', 'mo', 'di']:
            raise HTTPException(status_code=400, detail="Langue invalide. Utilisez: fr, mo, di")
        
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        deleted_count = rag.vector_store.delete_by_language(language)
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "rag_language_deleted",
            "admin",
            {"language": language, "deleted_count": deleted_count}
        )
        
        return {
            "success": True,
            "message": f"{deleted_count} documents en {language} supprimés du RAG",
            "language": language,
            "deleted_count": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur suppression langue RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/rag/source/{source}", tags=["Admin", "RAG"])
async def delete_rag_by_source(
    source: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Supprimer tous les documents d'une source spécifique"""
    try:
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        deleted_count = rag.vector_store.delete_by_source(source)
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "rag_source_deleted",
            "admin",
            {"source": source, "deleted_count": deleted_count}
        )
        
        return {
            "success": True,
            "message": f"{deleted_count} documents de la source '{source}' supprimés",
            "source": source,
            "deleted_count": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur suppression source RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/rag/clear-all", tags=["Admin", "RAG"])
async def clear_all_rag(
    confirm: bool = False,
    background_tasks: BackgroundTasks = None,
    _: bool = Depends(verify_admin)
):
    """⚠️ DANGER: Vider complètement la base RAG/FAISS (nécessite confirm=true)"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation requise. Ajoutez ?confirm=true pour confirmer la suppression totale"
            )
        
        if not rag or not rag.vector_store:
            raise HTTPException(status_code=503, detail="RAG non initialisé")
        
        stats_before = rag.vector_store.get_stats()
        total_docs = stats_before['total_documents']
        
        rag.vector_store.clear_all()
        
        # Log critique
        if background_tasks:
            background_tasks.add_task(
                db.add_admin_log,
                "rag_cleared_all",
                "admin",
                {"total_deleted": total_docs, "warning": "TOUTE LA BASE RAG VIDÉE"}
            )
        
        return {
            "success": True,
            "message": "⚠️ Toute la base RAG a été vidée",
            "documents_deleted": total_docs,
            "warning": "Cette action est irréversible"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur vidage RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/validate-and-ingest/{item_id}", tags=["Admin"])
async def validate_and_ingest_knowledge(
    item_id: str,
    action_data: dict,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Valider une connaissance expert et l'ingérer dans le RAG (admin uniquement)"""
    try:
        action = action_data.get("action")
        
        if action not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Action doit être 'approve' ou 'reject'")
        
        # Chercher la contribution dans MongoDB
        contribution = None
        
        # Chercher par ID direct
        contribution = db.contributions.find_one({"id": item_id})
        
        # Chercher dans validation_queue si pas trouvé (format: val_xxx)
        if not contribution and item_id.startswith("val_"):
            original_id = item_id[4:]
            contribution = db.validation_queue.find_one({"id": original_id})
        
        if not contribution:
            raise HTTPException(status_code=404, detail="Contribution non trouvée dans MongoDB")
        
        # Action: Rejeter
        if action == "reject":
            # Mettre à jour le statut dans MongoDB
            db.contributions.update_one(
                {"id": item_id},
                {"$set": {
                    "status": "rejected",
                    "validated_at": datetime.now(),
                    "validated_by": "admin"
                }}
            )
            
            # Log dans MongoDB
            background_tasks.add_task(
                db.add_admin_log,
                "contribution_rejected",
                "admin",
                {"item_id": item_id, "title": contribution.get("title")}
            )
            
            logger.info(f"Contribution rejetée dans MongoDB: {item_id}")
            
            return {
                "message": "Contribution rejetée",
                "item_id": item_id,
                "action": "rejected"
            }
        
        # Action: Approuver et ingérer
        if action == "approve":
            # Extraire le contenu
            content = contribution.get("content", "")
            title = contribution.get("title", "Sans titre")
            category = contribution.get("category", "general")
            
            if not content:
                raise HTTPException(status_code=400, detail="Contenu vide, impossible d'ingérer")
            
            # Ingérer dans le RAG
            try:
                text_with_title = f"{title}\n\n{content}"
                rag.ingest([text_with_title], f"expert-{contribution.get('expertName', 'unknown')}")
                logger.info(f"✅ Contenu ingéré dans RAG: {title}")
            except Exception as e:
                logger.error(f"Erreur ingestion RAG: {e}")
                raise HTTPException(status_code=500, detail=f"Erreur d'ingestion dans le RAG: {str(e)}")
            
            # Mettre à jour la contribution dans MongoDB
            db.contributions.update_one(
                {"id": item_id},
                {"$set": {
                    "status": "validated",
                    "validated_at": datetime.now(),
                    "validated_by": "admin",
                    "ingested_to_rag": True
                }}
            )
            
            # Ajouter le document dans MongoDB
            document_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{item_id}"
            new_document = {
                "id": document_id,
                "filename": f"{title}.txt",
                "category": category,
                "uploaded_at": datetime.now(),
                "uploaded_by": contribution.get("expertName", "expert"),
                "size": len(content),
                "description": f"Contribution expert validée: {title}",
                "status": "processed",
                "source": "expert_contribution"
            }
            
            db.add_document(new_document)
            
            # Log dans MongoDB
            background_tasks.add_task(
                db.add_admin_log,
                "contribution_validated_and_ingested",
                "admin",
                {
                    "item_id": item_id,
                    "title": title,
                    "document_id": document_id,
                    "rag_ingested": True
                }
            )
            
            logger.info(f"✅ Contribution validée et ingérée dans MongoDB + RAG: {item_id} → {document_id}")
            
            return {
                "message": "Contribution validée et ingérée dans le RAG avec succès",
                "item_id": item_id,
                "document_id": document_id,
                "action": "approved",
                "ingested_to_rag": True
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur validation/ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/ingest-excel", tags=["Admin"])
async def admin_ingest_excel(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    _: bool = Depends(verify_admin)
):
    """Ingérer des connaissances depuis un fichier Excel (admin uniquement)"""
    try:
        # Vérifier l'extension
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Le fichier doit être au format Excel (.xlsx ou .xls)")
        
        # Lire le fichier Excel
        import pandas as pd
        import io
        
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Vérifier les colonnes requises
        required_columns = ['Question/Titre', 'Réponse/Contenu', 'Catégorie']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {', '.join(missing_columns)}. Utilisez le template Excel fourni."
            )
        
        # Traiter chaque ligne
        ingested_count = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                title = str(row['Question/Titre']).strip()
                content = str(row['Réponse/Contenu']).strip()
                category = str(row['Catégorie']).strip()
                tags = str(row.get('Tags', '')).strip() if 'Tags' in row else ''
                
                # Ignorer les lignes vides
                if not title or not content or title == 'nan' or content == 'nan':
                    continue
                
                # Créer le texte complet pour ingestion
                full_text = f"{title}\n\n{content}"
                
                # Ingérer dans le RAG
                rag.ingest([full_text], f"admin-excel-{category}")
                
                # Sauvegarder dans MongoDB
                document_data = {
                    "id": f"excel_{datetime.now().strftime('%Y%m%d%H%M%S')}_{index}",
                    "filename": f"{title[:30]}.txt",
                    "title": title,
                    "content": content,
                    "category": category,
                    "tags": [tag.strip() for tag in tags.split(',') if tag.strip()],
                    "uploaded_at": datetime.now(),
                    "uploaded_by": "admin",
                    "source": "excel_import",
                    "status": "processed",
                    "size": len(content)
                }
                
                db.add_document(document_data)
                ingested_count += 1
                
            except Exception as e:
                error_msg = f"Ligne {index + 2}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Erreur ingestion ligne {index + 2}: {e}")
                continue
        
        # Log dans MongoDB
        if background_tasks:
            background_tasks.add_task(
                db.add_admin_log,
                "excel_import",
                "admin",
                {
                    "filename": file.filename,
                    "ingested_count": ingested_count,
                    "errors_count": len(errors)
                }
            )
        
        logger.info(f"✅ Excel importé: {ingested_count} connaissances ingérées, {len(errors)} erreurs")
        
        result = {
            "message": f"Import Excel terminé avec succès",
            "ingested_count": ingested_count,
            "total_rows": len(df),
            "errors_count": len(errors),
            "success": True
        }
        
        if errors:
            result["errors"] = errors[:10]  # Limiter à 10 erreurs pour la réponse
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur import Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'import Excel: {str(e)}")


@app.post("/api/admin/ingest-json", tags=["Admin"])
async def admin_ingest_json(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    _: bool = Depends(verify_admin)
):
    """Ingérer des connaissances multilingues depuis un fichier JSON (admin uniquement)"""
    try:
        # Vérifier l'extension
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Le fichier doit être au format JSON (.json)")
        
        # Lire le fichier JSON
        import json
        
        contents = await file.read()
        data = json.loads(contents.decode('utf-8'))
        
        # Vérifier que c'est un tableau
        if not isinstance(data, list):
            raise HTTPException(
                status_code=400,
                detail="Le JSON doit être un tableau de connaissances"
            )
        
        # Traiter chaque connaissance multilingue
        ingested_count = 0
        errors = []
        
        for index, item in enumerate(data):
            try:
                # Vérifier la structure
                if 'categorie' not in item or 'langues' not in item:
                    errors.append({
                        "index": index,
                        "error": "Structure invalide: 'categorie' et 'langues' requis"
                    })
                    continue
                
                category = item['categorie']
                langues = item['langues']
                
                # Traiter chaque langue (fr, mo, di, etc.)
                for lang_code, lang_data in langues.items():
                    try:
                        question = str(lang_data.get("question", "")).strip()
                        reponse_courte = str(lang_data.get("reponse_courte", "")).strip()
                        reponse_detaillee = str(lang_data.get("reponse_detaillee", "")).strip()
                        conseil = str(lang_data.get("conseil", "")).strip()
                        avertissement = str(lang_data.get("avertissement", "")).strip()

                        # Ne rien ingérer si absolument rien n'est rempli
                        # ⚠️ Ancien format (seulement "reponse") n'est plus supporté :
                        # on exige au moins question OU une des réponses enrichies.
                        if not any([question, reponse_courte, reponse_detaillee, conseil, avertissement]):
                            continue

                        lines = []

                        if question:
                            lines.append(f"Question: {question}")

                        # Format enrichi (obligatoire à présent)
                        if reponse_courte or reponse_detaillee or conseil or avertissement:
                            if reponse_courte:
                                lines.append(f"Idée principale: {reponse_courte}")
                            if reponse_detaillee:
                                lines.append(f"Explication: {reponse_detaillee}")
                            if conseil:
                                lines.append(f"Conseil pratique: {conseil}")
                            if avertissement:
                                lines.append(f"Avertissement: {avertissement}")

                        full_text = "\n".join(lines).strip()
                        if not full_text:
                            continue

                        # Choisir une réponse principale pour MongoDB (ancien champ "reponse" ignoré)
                        answer_for_db = reponse_detaillee or reponse_courte or ""

                        # Ingérer dans le RAG
                        if rag:
                            rag.ingest([full_text], f"admin-json-{category}-{lang_code}")

                        # Sauvegarder dans MongoDB (en conservant les champs enrichis si présents)
                        document_data = {
                            "id": f"json_{category}_{lang_code}_{index}",
                            "filename": f"{(question or (reponse_courte or answer_for_db or ''))[:50]}... ({lang_code})",
                            "description": f"Connaissance {category} en {lang_code}",
                            "category": category,
                            "language": lang_code,
                            "question": question,
                            "answer": answer_for_db,
                            "content": full_text,
                            "uploaded_at": datetime.now(),
                            "uploaded_by": "admin",
                            "source": "json_multilingual",
                            "size": len(full_text),
                            "status": "processed",
                        }

                        # Ajouter les champs enrichis pour exploitation future (facultatif)
                        if reponse_courte:
                            document_data["short_answer"] = reponse_courte
                        if reponse_detaillee:
                            document_data["detailed_answer"] = reponse_detaillee
                        if conseil:
                            document_data["advice"] = conseil
                        if avertissement:
                            document_data["warning"] = avertissement

                        db.add_document(document_data)
                        ingested_count += 1

                    except Exception as lang_error:
                        errors.append({
                            "index": index,
                            "language": lang_code,
                            "error": str(lang_error)
                        })
                        continue
                        
            except Exception as item_error:
                errors.append({
                    "index": index,
                    "error": str(item_error)
                })
                continue
        
        # Log dans MongoDB
        background_tasks.add_task(
            db.add_admin_log,
            "json_ingest",
            "admin",
            {
                "filename": file.filename,
                "ingested_count": ingested_count,
                "total_items": len(data),
                "errors_count": len(errors)
            }
        )
        
        logger.info(f"✅ JSON importé: {ingested_count} connaissances ingérées, {len(errors)} erreurs")
        
        result = {
            "message": f"Import JSON terminé avec succès",
            "ingested_count": ingested_count,
            "total_items": len(data),
            "errors_count": len(errors),
            "success": True
        }
        
        if errors:
            result["errors"] = errors[:10]  # Limiter à 10 erreurs pour la réponse
        
        return result
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Erreur parsing JSON: {e}")
        raise HTTPException(status_code=400, detail=f"JSON invalide: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur import JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'import JSON: {str(e)}")


@app.post("/api/admin/system/action", tags=["Admin"])
async def admin_system_action(
    action_data: SystemAction,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Actions système"""
    try:
        action = action_data.action.lower()
        
        if action == "restart":
            # Simuler un redémarrage
            background_tasks.add_task(
                db.add_admin_log,
                "system_restart",
                "admin",
                {"force": action_data.force}
            )
            logger.warning("Redémarrage système demandé")
            return {"message": "Redémarrage initié", "action": "restart"}
            
        elif action == "backup":
            # Créer une sauvegarde à partir des collections MongoDB
            backup_file = f"data/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            if db is None:
                raise HTTPException(status_code=500, detail="MongoDB non disponible pour la sauvegarde")

            snapshot = {
                "contributions": list(db.contributions.find({})),
                "validation_queue": list(db.validation_queue.find({})),
                "experts": list(db.experts.find({})),
                "admin_logs": list(db.admin_logs.find({})),
                "api_keys": list(db.api_keys.find({})),
                "system_stats": list(db.system_stats.find({})),
                "chat_conversations": list(db.chat_conversations.find({})),
                "chat_categories": list(db.chat_categories.find({})),
                "documents": list(db.documents.find({})),
                "notifications": list(db.notifications.find({})),
                "audit_logs": list(db.audit_logs.find({}))
            }

            # Sérialisation en JSON (ObjectId/datetime -> str)
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, default=str)

            background_tasks.add_task(
                db.add_admin_log,
                "system_backup",
                "admin",
                {"backup_file": backup_file}
            )

            logger.info(f"Sauvegarde créée: {backup_file}")
            return {"message": "Sauvegarde créée", "backup_file": backup_file}
            
        elif action == "clear_logs":
            # Nettoyer les logs dans MongoDB
            if db is not None:
                db.admin_logs.delete_many({})

            background_tasks.add_task(
                db.add_admin_log,
                "clear_logs",
                "admin",
                {}
            )

            logger.info("Logs nettoyés dans MongoDB")
            return {"message": "Logs nettoyés avec succès"}
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Action non supportée: {action}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur action système: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/logs", tags=["Admin"])
async def get_admin_logs(
    limit: Optional[int] = 100,
    _: bool = Depends(verify_admin)
):
    """Obtenir les logs système"""
    try:
        logs_file = "data/logs.json"
        
        if not os.path.exists(logs_file):
            return []
        
        with open(logs_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # Retourner les logs les plus récents en premier
        logs.reverse()
        
        return logs[:limit]
        
    except Exception as e:
        logger.error(f"Erreur get_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/health", response_model=HealthResponse, tags=["Public"])
async def api_health():
    """Health check endpoint pour l'API"""
    # Appelle la même fonction que /health
    return await health()

@app.post("/api/upload", response_model=UploadResponse, tags=["Public"])
async def upload_document_public(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(default="general"),
    description: Optional[str] = Form(None)
):
    """Uploader un document (endpoint public pour les utilisateurs)"""
    try:
        # Vérifier le type de fichier
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.png', '.jpeg']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non autorisé. Types acceptés: {', '.join(allowed_extensions)}"
            )
        
        # Générer un nom de fichier unique
        file_id = str(uuid.uuid4())[:8]
        original_name = os.path.splitext(file.filename)[0]
        safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        new_filename = f"{safe_name}_{file_id}{file_ext}"
        
        # Sauvegarder le fichier
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Obtenir la taille du fichier
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB"
        
        # PRÉPAREZ les données pour MongoDB
        document_record = {
            "id": f"doc_{file_id}",
            "filename": new_filename,
            "original_name": file.filename,
            "category": category,
            "description": description,
            "size": size_str,
            "uploaded_by": "public_user",
            "file_path": file_path
        }
        
        # AJOUT à MongoDB
        doc_id = db.add_document(document_record)
        
        # Log
        background_tasks.add_task(
            db.add_admin_log,
            "document_uploaded_public",
            "public_user",
            {
                "filename": new_filename,
                "category": category,
                "size": size_str
            }
        )
        
        logger.info(f"📄 Document public uploadé: {new_filename}")
        
        return UploadResponse(
            message="Document uploadé avec succès",
            filename=file.filename,
            size=size_str,
            uploaded_at=datetime.now().isoformat(),
            file_url=f"/uploads/{new_filename}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur upload document public: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")
    

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_ai(message: ChatMessage, background_tasks: BackgroundTasks):
    """Endpoint pour discuter avec l'IA (chat user) - VERSION INTELLIGENTE AVEC DÉTECTION DE LANGUE"""
    try:
        # ============ IMPORTER LE SERVICE CONVERSATIONNEL ============
        from ai.service.conversation import ConversationService
        conversation_service = ConversationService()
        
        logger.info(f"🤖 Question reçue: {message.message[:100]}...")
        
        # ============ ANALYSE INTELLIGENTE DU MESSAGE ============
        # Détection de langue préliminaire
        detected_lang = conversation_service.detect_language(message.message)
        logger.info(f"🌍 Langue détectée: {detected_lang}")
        
        # ============ APPEL AU RAG AVEC FILTRE LANGUE ============
        response_text, sources = rag.ask(message.message, k=5, language=detected_lang)
        
        # ============ ANALYSE ET FORMATAGE CONVERSATIONNEL ============
        analysis = conversation_service.analyze_and_respond(
            user_message=message.message,
            raw_rag_answer=response_text,
            category=message.category
        )
        
        # Utiliser la réponse formatée intelligemment
        final_response = analysis['response']
        intent = analysis['intent']
        needs_clarification = analysis['needs_clarification']
        
        logger.info(f"💬 Intent: {intent}, Langue: {detected_lang}, Clarification: {needs_clarification}")
        
        # Extraire les noms de sources depuis les documents
        source_names = []
        if sources:
            # Limiter à 3 sources les plus pertinentes
            source_lines = sources.split('\n')[:3]
            for line in source_lines:
                if line.strip():
                    # Extraire juste la question/titre (première ligne)
                    if '\n\n' in line:
                        title = line.split('\n\n')[0]
                    else:
                        title = line[:80] + "..." if len(line) > 80 else line
                    source_names.append(title.strip())
        
        if not source_names:
            source_names = [f"Base de connaissances IA Souveraine ({detected_lang.upper()})"]
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"✅ Réponse IA générée en {detected_lang} avec {len(source_names)} sources")
        
        # ============ SAUVEGARDE MONGODB ============
        conversation_data = {
            "user_message": message.message,
            "ai_response": final_response,
            "category": message.category,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(),
            "language": detected_lang,
            "intent": intent,
            "needs_clarification": needs_clarification,
            "sources": source_names,
            "user_ip": "unknown"
        }
        
        # SAUVEGARDE DANS MONGODB
        from mongodb import db
        mongo_id = db.save_chat_conversation(conversation_data)
        
        # Log admin
        background_tasks.add_task(
            db.add_admin_log,
            "chat_conversation",
            "system",
            {
                "conversation_id": conversation_id,
                "mongo_id": mongo_id,
                "category": message.category,
                "language": detected_lang,
                "intent": intent,
                "message_length": len(message.message)
            }
        )
        
        logger.info(f"💬 Conversation sauvegardée MongoDB ID: {mongo_id}")
        # ============ FIN SAUVEGARDE ============
        
        return ChatResponse(
            response=final_response,
            confidence=0.92,
            sources=source_names,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories", response_model=List[Category], tags=["Chat"])
async def get_chat_categories():
    """Obtenir les catégories pour le chat"""
    try:
        # RÉCUPÉRATION depuis MongoDB
        categories = db.get_chat_categories()
        return categories
        
    except Exception as e:
        logger.error(f"Erreur categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# SERVEUR ET ÉVÉNEMENTS
# ============================================

# Serveur de fichiers statiques
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Serveur d'audio pour TTS (mooré, dioula)
import os
audio_dir = os.path.join(os.path.dirname(__file__), "audio")
if os.path.exists(audio_dir):
    app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")
    logger.info(f"🎵 Serveur audio monté: /audio")
else:
    logger.warning(f"⚠️ Dossier audio non trouvé: {audio_dir}")

# Événements de démarrage/arrêt
# Commenté temporairement pour debug
# @app.on_event("startup")
# async def startup_event():
#     """Événement de démarrage de l'application"""
#     logger.info("🚀 Démarrage de l'API Expert...")
#     logger.info(f"✅ API Expert prête sur http://localhost:8000")
#     logger.info(f"📚 Documentation: http://localhost:8000/docs")
#     logger.info(f"🔑 Token Expert: {EXPERT_KEY}")
#     logger.info(f"💬 Chat user disponible sur /api/chat")
#     
#     # Essayer de se connecter à MongoDB (non bloquant)
#     try:
#         db_name = db.db_name
#         logger.info(f"🗄️  MongoDB connecté: {db_name}")
#     except Exception as e:
#         logger.warning(f"⚠️  MongoDB non disponible: {e}")
#         logger.info("💡 L'API continuera de fonctionner sans MongoDB")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Événement d'arrêt de l'application"""
#     logger.info("👋 Arrêt de l'API Expert...")


if __name__ == "__main__":
    import uvicorn
    logger.info("="*60)
    logger.info("🚀 DÉMARRAGE BACKEND YINGRE AI")
    logger.info("="*60)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Désactiver reload pour éviter les problèmes
        log_level="info",
        lifespan="off"  # Désactiver lifespan
    )