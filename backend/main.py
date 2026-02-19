from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from db import init_db, create_user, get_user
from fastapi import UploadFile, File, Form
import shutil
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from jose import JWTError, jwt
from db import save_chat, get_recent_chats
import sqlite3
from db import get_recent_chats
from fastapi.staticfiles import StaticFiles
import ollama
import re


from rag_engine import load_all_pdfs, create_index, ask_question, interpret_image_with_vision, highlight_diagram_elements, analyze_blueprint_component

# Session storage for tracking user's last shown images
user_image_sessions: Dict[str, List[str]] = {}


app = FastAPI()

app.mount("/extracted_images", StaticFiles(directory="extracted_images"), name="images")


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
SECRET_KEY = "change_this_secret_to_a_secure_value"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

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
    last_image: Optional[str] = None  # ⭐ NEW: Track the last shown image
def ask(req: AskRequest, current_user: dict = Depends(get_current_user)):
    username = current_user.get("username")
    user_role = current_user.get("role")
    role_to_query = req.role if user_role in ["ADMIN", "CAPTAIN"] else user_role
    
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
            # Analyze the specific image the user was asking about
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
            
            save_chat(username, req.question, image_analysis.get("interpretation", ""))
            
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
                "analysis_type": "image_analysis"
            }
    
    # ========== NORMAL DOCUMENT QUERY ==========
    filtered_chunks = []
    filtered_sources = []
    filtered_pages = []

    for chunk, r, src, page in zip(chunks, chunk_roles, chunk_sources, chunk_pages):
        if r == role_to_query or role_to_query == "ADMIN":
            filtered_chunks.append(chunk)
            filtered_sources.append(src)
            filtered_pages.append(page)

    print("MATCHED:", list(zip(filtered_sources[:5], filtered_pages[:5])))
    temp_index = create_index(filtered_chunks)

    # 🧠 MEMORY PART STARTS HERE
    history = get_recent_chats(username, limit=5)

    # Get answer AND the indices of matched chunks
    answer, matched_indices = ask_question(
        req.question,
        temp_index,
        filtered_chunks,
        history,
        return_indices=True
    )

    save_chat(username, req.question, answer)
    # 🧠 MEMORY PART ENDS HERE
    
    # Extract pages and sources from the MATCHED chunks (not all filtered chunks)
    relevant_pages = []
    relevant_sources = []
    
    for idx in matched_indices:
        relevant_pages.append(filtered_pages[idx])
        relevant_sources.append(filtered_sources[idx])
    
    # Get images from the relevant pages only
    candidate_images = []
    
    # Build prefixes from the actual matched chunk pages
    image_prefixes = set()
    for src, page in zip(relevant_sources, relevant_pages):
        prefix = f"{src}_page{page}_"
        image_prefixes.add(prefix)
    
    # Get all available images
    if os.path.exists("extracted_images"):
        for img in os.listdir("extracted_images"):
            # Check if image starts with any of the relevant prefixes
            for prefix in image_prefixes:
                if img.startswith(prefix):
                    candidate_images.append(img)
                    break
    
    # Remove duplicates
    candidate_images = list(set(candidate_images))
    
    # Extract rule/section references from the question and answer
    # Extract RULE numbers mentioned in the question (e.g., "RULE 25", "RULE 27")
    rule_references = set()
    
    # Search in question
    rule_matches = re.findall(r'RULE\s+(\d+)', req.question, re.IGNORECASE)
    rule_references.update(rule_matches)
    
    # Search in answer
    rule_matches = re.findall(r'RULE\s+(\d+)', answer, re.IGNORECASE)
    rule_references.update(rule_matches)
    
    # Also search for section references like "25(c)", "27(a)", etc
    section_matches = re.findall(r'(?:RULE\s+)?(\d+)\s*\(', answer, re.IGNORECASE)
    rule_references.update(section_matches)
    
    related_images = []
    
    # If we found specific rule references, use them to filter images
    if rule_references and candidate_images:
        # Get the PDF text for the pages to check which rules are on those pages
        from rag_engine import load_pdf_text
        
        for img in candidate_images:
            # Extract source PDF and page number from image filename
            # Format: PDF_name_pageN_index.ext
            parts = img.rsplit('_', 2)
            if len(parts) >= 3:
                page_num = parts[1].replace('page', '')
                try:
                    page_num = int(page_num)
                    
                    # Find which PDF this image belongs to
                    pdf_source = None
                    for src in set(filtered_sources):
                        if src in img:
                            pdf_source = src
                            break
                    
                    if pdf_source:
                        # Load the PDF and get text from that page
                        pdf_path = os.path.join("../documents", pdf_source)
                        if os.path.exists(pdf_path):
                            from pypdf import PdfReader
                            reader = PdfReader(pdf_path)
                            if page_num < len(reader.pages):
                                page_text = reader.pages[page_num].extract_text()
                                
                                # Check if any of the referenced rules are in this page
                                for rule_ref in rule_references:
                                    if re.search(rf'RULE\s+{rule_ref}(?:\s|[^0-9]|$)', page_text, re.IGNORECASE):
                                        related_images.append(img)
                                        break
                except:
                    # If extraction fails, include the image (better to have it than not)
                    related_images.append(img)
    
    # Fallback: if no images were matched by rule reference, include all candidates
    if not related_images and candidate_images:
        related_images = candidate_images
    
    # Interpret images with LLaVA vision model
    image_interpretations = []
    if related_images:
        for img in related_images:
            img_path = os.path.join("extracted_images", img)
            if os.path.exists(img_path):
                print(f"Interpreting image: {img}")
                interpretation = interpret_image_with_vision(img_path, req.question)
                image_interpretations.append({
                    "image": img,
                    "interpretation": interpretation
                })
    
    # Store shown images in user session for future reference
    store_images_for_user(username, related_images)

    return {
        "answer": answer,
        "source": list(set(filtered_sources)),
        "images": related_images,
        "image_details": image_interpretations,
        "analysis_type": "document_query"
    }


@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    username = current_user.get("username")

    chats = get_recent_chats(username, limit=20)

    return {
        "history": [
            {"question": q, "answer": a}
            for q, a in chats
        ]
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
