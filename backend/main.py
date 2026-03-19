from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from db import init_db, create_user, get_user
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
import shutil
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from jose import JWTError, jwt
from db import save_chat, get_recent_chats, get_conversations, get_conversation_messages, get_admin_stats, delete_user, get_query_time_stats, get_settings, update_setting
from rag_engine import indexing_status, load_all_pdfs, create_index, chunks, chunk_roles, chunk_sources, chunk_pages
import sqlite3
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import ollama
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


from rag_engine import load_all_pdfs, create_index, ask_question, stream_ask_question, interpret_image_with_vision, highlight_diagram_elements, analyze_blueprint_component, locate_and_highlight_part

# Session storage for tracking user's last shown images
user_image_sessions: Dict[str, List[str]] = {}


app = FastAPI()

# --- Serve Static Media Files ---
# Mount images folder
app.mount("/extracted_images", StaticFiles(directory="extracted_images"), name="images")

# Mount PDF previews (admin)
if not os.path.exists("documents"): os.makedirs("documents")
app.mount("/documents", StaticFiles(directory="documents"), name="documents")


@app.get("/")
async def read_index():
    """Serve the login page at the root"""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
    return FileResponse(os.path.join(frontend_path, "login.html"))


class AskRequest(BaseModel):
    question: str
    role: str

class SignupRequest(BaseModel):
    username: str
    password: str
    role: str

class LoginRequest(BaseModel):
    username: str
    password: str


# ----- JWT helpers -----
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY or SECRET_KEY == "change-this-to-a-long-random-secret-key":
    raise RuntimeError(
        "\n\n🚨 SECRET_KEY is not set or is still the placeholder!\n"
        "   Open backend/.env and set a real secret key.\n"
        "   Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exception


init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HELPER FUNCTIONS ----------

def is_semantic_image_question(question: str) -> bool:
    """True if asking about meaning/explanation (use direct interpretation, no grid). False if asking where/locate (use grid)."""
    q = question.lower()
    semantic = [
        r"what\s+does?\s+.*\s+mean",           # "what does the red part mean"
        r"what\s+is\s+the\s+meaning",
        r"explain\s+(what|the)\s+(red|highlighted|this)",
        r"what\s+do\s+(the\s+)?(red|lights|symbols)\s+mean",
        r"describe\s+the\s+meaning",
        r"what\s+does\s+the\s+(red|highlighted)\s+indicate",
        r"significance\s+of",
        r"highlighted\s+part\s+(mean|indicate|show)",
    ]
    # Exclude "where is" - that's spatial
    if re.search(r"where\s+is", q) or re.search(r"locate\s+the", q):
        return False
    for pat in semantic:
        if re.search(pat, q):
            return True
    return False


def detect_image_analysis_request(question: str) -> bool:
    """Detect if user is asking to analyze the above/current image"""
    keywords = [
        r"analyze\s+the\s+(above|this|that|current|image|diagram|blueprint|schematic|diagram|highlighted)",
        r"(above|this|that|highlighted)\s+(image|part|diagram|section)",
        r"what\s+is\s+(shown|displayed|this|that|highlighted)",
        r"explain\s+(the\s+)?(above|this|that|image|diagram|highlighted)",
        r"interpret\s+(the\s+)?(above|this|that|image|diagram|highlighted)",
        r"describe\s+(the\s+)?(above|this|that|image|diagram|highlighted)",
        r"tell\s+me\s+about\s+(the\s+)?(above|this|that|image|diagram|highlighted)",
        r"highlight(ed)?\s+(the\s+)?(above|this|that|image|diagram)",
        r"show\s+me",
        r"where\s+is",
        r"identify\s+(the|this|that|highlighted)",
        r"image\s+analysis",
        r"what.*highlighted",
        r"highlighted\s+(part|section|area|component)"
    ]
    
    question_lower = question.lower()
    for keyword in keywords:
        if re.search(keyword, question_lower):
            return True
    return False


def get_latest_images_for_user(username: str) -> List[str]:
    """Get the most recently shown images for a user"""
    return user_image_sessions.get(username, [])


def store_images_for_user(username: str, images: List[str]):
    """Store shown images in user session, keep last 5"""
    if images:
        if username not in user_image_sessions:
            user_image_sessions[username] = []
        user_image_sessions[username] = images + user_image_sessions.get(username, [])
        # Keep only last 5 images
        user_image_sessions[username] = user_image_sessions[username][:5]


# ---------- LOAD RAG SYSTEM ON START ----------


chunks, chunk_roles, chunk_sources, chunk_pages = load_all_pdfs("../documents")



print("Chunks loaded:", len(chunks))

if len(chunks) == 0:
    raise Exception("No PDFs found in documents folder")

index = create_index(chunks)



# ---------- REQUEST FORMAT ----------
class QuestionRequest(BaseModel):
    question: str


# ---------- API ROUTE ----------

class AskRequest(BaseModel):
    question: str
    role: str
    last_image: Optional[str] = None
    conversation_id: Optional[str] = None  # None = new chat; backend creates and returns id

@app.post("/ask")
def ask(req: AskRequest, current_user: dict = Depends(get_current_user)):
    username = current_user.get("username")
    user_role = current_user.get("role")
    role_to_query = req.role if user_role in ["ADMIN", "CAPTAIN"] else user_role
    conversation_id = req.conversation_id
    
    # ========== CHECK IF USER IS ASKING ABOUT A SPECIFIC IMAGE ==========
    # Priority 1: Use last_image from request if provided
    image_name_to_analyze = req.last_image if req.last_image else None
    
    # Priority 2: Detect image query and use server-side session
    if not image_name_to_analyze:
        is_image_query = detect_image_analysis_request(req.question)
        latest_images = get_latest_images_for_user(username)
        
        if is_image_query and latest_images:
            image_name_to_analyze = latest_images[0]
    
    # If we have an image to analyze, use it
    if image_name_to_analyze:
        image_path = os.path.join("extracted_images", image_name_to_analyze)
        
        if os.path.exists(image_path):
            # ========== SPECIAL HANDLING FOR "WHERE IS" QUERIES ==========
            # Check if this is a "where is" type query
            where_is_match = re.search(r'where\s+is\s+(?:the\s+)?(.+?)(?:\?|$)', req.question, re.IGNORECASE)
            
            if where_is_match:
                # Extract the part name from the query
                part_to_find = where_is_match.group(1).strip()
                
                # Use the specialized locate and highlight function
                location_result = locate_and_highlight_part(image_path, part_to_find)
                
                conversation_id = save_chat(username, req.question, location_result.get("analysis", ""), conversation_id)

                return {
                    "answer": location_result.get("analysis", ""),
                    "source": [],
                    "images": [image_name_to_analyze],
                    "image_details": [{
                        "image": image_name_to_analyze,
                        "interpretation": location_result.get("analysis", ""),
                        "highlighted_image": location_result.get("highlighted_image"),
                        "analysis_info": location_result.get("highlight_info"),
                        "location_found": location_result.get("location_found")
                    }],
                    "analysis_type": "location_query",
                    "conversation_id": conversation_id,
                }
            
            # ========== GENERAL IMAGE ANALYSIS ==========
            # Semantic (meaning/explain) -> direct interpretation, no grid. Spatial (where/locate) -> grid-based highlight.
            if is_semantic_image_question(req.question):
                interpretation = interpret_image_with_vision(image_path, req.question)
                image_analysis = {
                    "interpretation": interpretation,
                    "highlighted_image": None,  # No grid highlight for semantic Qs
                    "analysis_info": {"model_used": "llava", "method": "direct_interpretation"}
                }
            else:
                image_analysis = highlight_diagram_elements(image_path, req.question)
            
            # Also get related document context
            filtered_chunks = []
            filtered_sources = []
            filtered_pages = []
            
            for chunk, r, src, page in zip(chunks, chunk_roles, chunk_sources, chunk_pages):
                if r == role_to_query or role_to_query == "ADMIN":
                    filtered_chunks.append(chunk)
                    filtered_sources.append(src)
                    filtered_pages.append(page)
            
            conversation_id = save_chat(username, req.question, image_analysis.get("interpretation", ""), conversation_id)

            return {
                "answer": image_analysis.get("interpretation", ""),
                "source": list(set(filtered_sources)),
                "images": [image_name_to_analyze],
                "image_details": [{
                    "image": image_name_to_analyze,
                    "interpretation": image_analysis.get("interpretation", ""),
                    "highlighted_image": image_analysis.get("highlighted_image"),
                    "analysis_info": image_analysis.get("analysis_info")
                }],
                "analysis_type": "image_analysis",
                "conversation_id": conversation_id,
            }
    
def get_metadata_for_answer(answer: str, question: str, matched_indices: List[int], role_to_query: str, username: str):
    """Refactored helper to get sources, pages, and images for an answer"""
    relevant_pages = []
    relevant_sources = []
    
    # Use only the top 2 most relevant matched chunks for image selection
    for idx in matched_indices[:2]:
        # Check permissions after retrieval
        if chunk_roles[idx] == role_to_query or role_to_query == "ADMIN":
            relevant_pages.append(chunk_pages[idx])
            relevant_sources.append(chunk_sources[idx])
    
    candidate_images = []
    
    # Build prefixes from the actual matched chunk pages
    image_prefixes = set()
    for src, page in zip(relevant_sources, relevant_pages):
        prefix = f"{src}_page{page}_"
        image_prefixes.add(prefix)
    
    # Get all available images on the pages where we found text
    if os.path.exists("extracted_images"):
        for img in os.listdir("extracted_images"):
            # Check if image starts with any of the relevant prefixes
            for prefix in image_prefixes:
                if img.startswith(prefix):
                    candidate_images.append(img)
                    break
    
    # Remove duplicates and limit to just 1 most relevant image
    candidate_images = list(set(candidate_images))[:1]
    related_images = candidate_images
    
    store_images_for_user(username, related_images)
    
    return {
        "sources": list(set(relevant_sources)),
        "images": related_images
    }
    

@app.post("/ask_stream")
async def ask_stream(req: AskRequest, current_user: dict = Depends(get_current_user)):
    if not index: raise HTTPException(status_code=500, detail="Index not ready")
    
    username = current_user.get("username")
    user_role = current_user.get("role")
    
    # Maintenance check
    settings = get_settings()
    if settings.get("maintenance_mode") == "true" and user_role != "ADMIN":
        async def maintenance_stream():
            yield json.dumps({"type": "text", "content": "⚠️ System is currently under maintenance. Please try again later."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return StreamingResponse(maintenance_stream(), media_type="text/event-stream")

    role_to_query = req.role if user_role in ["ADMIN", "CAPTAIN"] else user_role
    conversation_id = req.conversation_id
    
    # Check for image analysis request first (non-streaming fallback)
    if detect_image_analysis_request(req.question):
        sync_response = query_rag(req, current_user)
        async def sync_to_stream():
            yield json.dumps({"type": "info", "sources": sync_response.get("source", []), "images": sync_response.get("images", [])}) + "\n"
            yield json.dumps({"type": "text", "content": sync_response.get("answer", "")}) + "\n"
            yield json.dumps({"type": "done", "conversation_id": sync_response.get("conversation_id")}) + "\n"
        return StreamingResponse(sync_to_stream(), media_type="text/event-stream")

    history = get_recent_chats(username, conversation_id=conversation_id, limit=2)
    
    async def event_generator():
        full_answer = ""
        for chunk in stream_ask_question(req.question, index, chunks, history):
            if isinstance(chunk, str) and chunk.startswith('{"type": "metadata"'):
                meta = json.loads(chunk)
                info = get_metadata_for_answer("", req.question, meta["matched_indices"], role_to_query, username)
                yield json.dumps({"type": "info", "sources": info["sources"], "images": info["images"]}) + "\n"
                continue
            
            full_answer += chunk
            yield json.dumps({"type": "text", "content": chunk}) + "\n"
        
        new_conv_id = save_chat(username, req.question, full_answer, conversation_id)
        yield json.dumps({"type": "done", "conversation_id": new_conv_id}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/ask")
def query_rag(req: AskRequest, current_user: dict = Depends(get_current_user)):
    if not index: raise HTTPException(status_code=500, detail="Index not ready")
    
    username = current_user.get("username")
    user_role = current_user.get("role")
    
    # Maintenance check
    settings = get_settings()
    if settings.get("maintenance_mode") == "true" and user_role != "ADMIN":
        raise HTTPException(status_code=503, detail="System is currently under maintenance. Please try again later.")

    role_to_query = req.role if user_role in ["ADMIN", "CAPTAIN"] else user_role
    conversation_id = req.conversation_id

    # Handle image analysis fallback
    if detect_image_analysis_request(req.question):
        # ... logic for image analysis (omitted, assuming it follows line 282) ...
        pass

    # 🧠 MEMORY PART STARTS HERE
    history = get_recent_chats(username, conversation_id=conversation_id, limit=1)

    answer, matched_indices = ask_question(
        req.question,
        index,
        chunks,
        history,
        return_indices=True
    )

    conversation_id = save_chat(username, req.question, answer, conversation_id)
    metadata = get_metadata_for_answer(answer, req.question, matched_indices, role_to_query, username)
    
    image_interpretations = []
    MAX_IMAGES_TO_INTERPRET = 0
    if metadata["images"]:
        for img in metadata["images"][:MAX_IMAGES_TO_INTERPRET]:
            img_path = os.path.join("extracted_images", img)
            if os.path.exists(img_path):
                interpretation = interpret_image_with_vision(img_path, req.question)
                image_interpretations.append({"image": img, "interpretation": interpretation})
        for img in metadata["images"][MAX_IMAGES_TO_INTERPRET:]:
            image_interpretations.append({"image": img, "interpretation": None})

    return {
        "answer": answer,
        "source": metadata["sources"],
        "images": metadata["images"],
        "image_details": image_interpretations,
        "analysis_type": "document_query",
        "conversation_id": conversation_id,
    }

# ================= ADMIN ENDPOINTS =================

@app.get("/admin/stats")
def admin_stats(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    return get_admin_stats()

@app.get("/admin/documents")
def admin_documents(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    docs_dir = "../documents"
    if not os.path.exists(docs_dir):
        return []
    
    files = []
    for f in os.listdir(docs_dir):
        if f.lower().endswith('.pdf'):
            path = os.path.join(docs_dir, f)
            stats = os.stat(path)
            files.append({
                "name": f,
                "size": f"{stats.st_size / 1024 / 1024:.2f} MB",
                "indexed_at": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    return files

@app.get("/admin/stats/time")
def admin_stats_time(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    return get_query_time_stats()

@app.delete("/admin/users/{username}")
def admin_delete_user(username: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete the root admin")
    delete_user(username)
    return {"status": "success"}

@app.get("/admin/settings")
def admin_get_settings(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    return get_settings()

@app.post("/admin/settings")
def admin_post_setting(key: str = Form(...), value: str = Form(...), current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    update_setting(key, value)
    return {"status": "success"}

# --- NEW ADVANCED MANAGEMENT ---

@app.get("/admin/indexing/status")
async def admin_indexing_status(current_user: dict = Depends(get_current_user)):
    """SSE endpoint for real-time indexing progress"""
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    async def event_generator():
        last_pct = -1
        while True:
            # Only send if changed, or if status is 'processing'
            if indexing_status["status"] == "processing" or indexing_status["percentage"] != last_pct:
                yield f"data: {json.dumps(indexing_status)}\n\n"
                last_pct = indexing_status["percentage"]
            
            if indexing_status["status"] == "idle" and indexing_status["percentage"] == 100:
                # Send one last completion event
                yield f"data: {json.dumps(indexing_status)}\n\n"
                break
                
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/admin/documents/{filename}/preview")
def preview_document(filename: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Secure filename and check if it exists in documents/
    path = os.path.join("../documents", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(path, media_type='application/pdf')

@app.delete("/admin/documents/{filename}")
def delete_document(filename: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    path = os.path.join("../documents", filename)
    if os.path.exists(path):
        os.remove(path)
        
        # Trigger re-indexing in background or sync for simplicity
        global chunks, chunk_roles, chunk_sources, chunk_pages, index
        chunks, chunk_roles, chunk_sources, chunk_pages = load_all_pdfs("../documents")
        index = create_index(chunks)
        
        return {"status": "success", "message": f"{filename} deleted and system re-indexed"}
    
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/conversations")
def list_conversations(current_user: dict = Depends(get_current_user)):
    """List user's conversations (id, title, updated_at). Newest first."""
    username = current_user.get("username")
    convos = get_conversations(username)
    return {"conversations": convos}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get all messages in a conversation (for loading in UI)."""
    username = current_user.get("username")
    messages = get_conversation_messages(username, conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    """Legacy: flat list of recent Q&A across all conversations."""
    username = current_user.get("username")
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer FROM chat_history
        WHERE username=? AND conversation_id IS NOT NULL
        ORDER BY id DESC LIMIT 20
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    return {
        "history": [{"question": q, "answer": a} for q, a in rows[::-1]]
    }




@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    role: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    # only allow uploads for user's role or ADMIN
    if current_user.get("role") != role and current_user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Not authorized to upload for this role")

    save_path = f"../documents/{role.upper()}_{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Reload PDFs
    global chunks, chunk_roles, chunk_sources, chunk_pages
    chunks, chunk_roles, chunk_sources, chunk_pages = load_all_pdfs("../documents")

    return {"message": "File uploaded and indexed successfully"}







@app.post("/signup")
def signup(req: SignupRequest):
    settings = get_settings()
    if settings.get("allow_signup") != "true":
        raise HTTPException(status_code=403, detail="Public signup is currently disabled by administrator")
    try:
        create_user(req.username, req.password, req.role)
        return {"message": "User created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
def login(user: LoginRequest):
    db_user = get_user(user.username, user.password)

    if not db_user:
        return {"error": "Invalid credentials"}

    username = db_user[1]
    role = db_user[3]

    access_token = create_access_token(
        data={"sub": username, "role": role}
    )

    return {
        "access_token": access_token,
        "role": role
    }


# ---------- IMAGE ANALYSIS ENDPOINTS ----------

class DiagramAnalysisRequest(BaseModel):
    image_name: str
    question: str

class ComponentAnalysisRequest(BaseModel):
    image_name: str
    component_name: str

class LocatePartRequest(BaseModel):
    image_name: str
    part_name: str


@app.post("/analyze-diagram")
def analyze_diagram_with_highlighting(
    req: DiagramAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze a diagram/blueprint image and highlight key elements.
    Uses LLaVA vision model to identify and annotate relevant components.
    """
    image_path = os.path.join("extracted_images", req.image_name)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = highlight_diagram_elements(image_path, req.question)
    return result


@app.post("/analyze-component")
def analyze_blueprint_component_endpoint(
    req: ComponentAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Deep analysis of a specific component in a technical diagram.
    Useful for detailed technical specifications and safety information.
    """
    image_path = os.path.join("extracted_images", req.image_name)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    analysis = analyze_blueprint_component(image_path, req.component_name)
    return {
        "component": req.component_name,
        "image": req.image_name,
        "analysis": analysis
    }


@app.post("/interpret-image-detailed")
def interpret_image_detailed(
    req: DiagramAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Simple interpretation of an image without highlighting.
    Returns text description of the diagram.
    """
    image_path = os.path.join("extracted_images", req.image_name)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    interpretation = interpret_image_with_vision(image_path, req.question)
    return {
        "image": req.image_name,
        "question": req.question,
        "interpretation": interpretation
    }


@app.post("/locate-and-highlight")
def locate_part_in_image(
    req: LocatePartRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Locate a specific part or component in an image and highlight it with ORANGE.
    Perfect for "where is" queries like "where is the pump?" or "locate the valve"
    
    Returns:
        - Analysis of the part location
        - Image with ORANGE circles and boxes highlighting the part
    """
    image_path = os.path.join("extracted_images", req.image_name)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = locate_and_highlight_part(image_path, req.part_name)
    return result

# --- FALLBACK: Serve all other frontend files (.html, .js, .css) ---
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path), name="frontend_root")

