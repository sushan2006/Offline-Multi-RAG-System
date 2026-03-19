import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import ollama
import fitz
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import json
import torch

# ----------- GPU SETUP -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)} — running on CUDA")
else:
    print("⚠️  No GPU found — running on CPU")

print(f"Loading embedding model on {device}...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# ----------- PROGRESS TRACKING -----------
indexing_status = {"percentage": 0, "status": "idle", "current_file": "", "total_files": 0, "processed_files": 0}

# ⭐ GLOBAL STATE FOR CHUNKS
chunks = []
chunk_roles = []
chunk_sources = []
chunk_pages = []


# ----------- READ PDF -----------
import os

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text


def load_all_pdfs(folder):
    global indexing_status, chunks, chunk_roles, chunk_sources, chunk_pages

    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    total_files = len(files)
    indexing_status["total_files"] = total_files
    indexing_status["status"] = "processing"
    indexing_status["processed_files"] = 0
    indexing_status["percentage"] = 0

    for i, file in enumerate(files):
        path = os.path.join(folder, file)
        indexing_status["current_file"] = file
        indexing_status["percentage"] = int((i / total_files) * 100)
        
        extract_images_from_pdf(path, file)

        role = file.split("_")[0].upper()

        reader = PdfReader(path)
        
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages):
            # Update sub-progress for pages
            page_pct = int(((i + (page_num / total_pages)) / total_files) * 100)
            indexing_status["percentage"] = page_pct
            
            content = page.extract_text()

            if not content:
                continue

            parts = split_text(content)

            for part in parts:
                chunks.append(part)
                chunk_roles.append(role)
                chunk_sources.append(file)
                chunk_pages.append(page_num)   # ⭐ STORE PAGE

        indexing_status["processed_files"] += 1

    indexing_status["status"] = "idle"
    indexing_status["percentage"] = 100
    return chunks, chunk_roles, chunk_sources, chunk_pages


def extract_images_from_pdf(pdf_path, pdf_name):
    images = []

    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            img_name = f"{pdf_name}_page{page_index}_{img_index}.{ext}"
            img_path = os.path.join("extracted_images", img_name)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            images.append(img_name)

    return images



# ----------- SPLIT TEXT INTO CHUNKS -----------
def split_text(text, chunk_size=150):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks



# ----------- CREATE VECTOR INDEX -----------
def create_index(chunks):
    print(f"Encoding {len(chunks)} chunks on {device}...")
    # batch_size=64 keeps GPU memory usage low while still being fast
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    # Try to use GPU-accelerated FAISS if available
    cpu_index = faiss.IndexFlatL2(dimension)
    try:
        if device == "cuda" and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            index.add(np.array(embeddings, dtype='float32'))
            print("⚡ FAISS index placed on GPU")
        else:
            raise RuntimeError("faiss-gpu not available")
    except Exception:
        cpu_index.add(np.array(embeddings, dtype='float32'))
        index = cpu_index
        if device == "cuda":
            print("✅ Embeddings on GPU, FAISS index on CPU (install faiss-gpu for full GPU FAISS)")

    return index


# ----------- ASK QUESTION -----------
def ask_question(question, index, chunks, history=None, return_indices=False):
    """Sync version of ask_question (for backward compatibility)"""
    q_embedding = model.encode([question], convert_to_numpy=True)
    D, I = index.search(np.array(q_embedding, dtype='float32'), k=2)

    context = ""
    matched_indices = I[0].tolist()
    for i in matched_indices:
        context += chunks[i] + "\n"

    history_text = ""
    if history:
        for q, a in history:
            history_text += f"User: {q}\nAssistant: {a}\n"

    try:
        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Previous Conversation:\n{history_text}\n\nContext:\n{context}\n\nQuestion: {question}"}
            ],
            options={"num_predict": 256, "temperature": 0.1}
        )
        answer = response['message']['content']
    except Exception as e:
        answer = f"Error: {str(e)}"

    if return_indices:
        return answer, matched_indices
    return answer

# ----------- STREAM ASK QUESTION -----------
def stream_ask_question(question, index, chunks, history=None):
    """Generator version of ask_question that yields word-by-word chunks"""
    q_embedding = model.encode([question], convert_to_numpy=True)
    D, I = index.search(np.array(q_embedding, dtype='float32'), k=2)

    context = ""
    matched_indices = I[0].tolist()
    for i in matched_indices:
        context += chunks[i] + "\n"

    history_text = ""
    if history:
        for q, a in history:
            history_text += f"User: {q}\nAssistant: {a}\n"

    # yield initial metadata as the first chunk (JSON format)
    # This lets the frontend know which sources/indices were used immediately
    yield json.dumps({"type": "metadata", "matched_indices": matched_indices}) + "\n"

    try:
        # stream=True returns an iterable of response chunks
        stream = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Previous Conversation:\n{history_text}\n\nContext:\n{context}\n\nQuestion: {question}"}
            ],
            options={"num_predict": 512, "temperature": 0.1},
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
                
    except Exception as e:
        yield f"\n\n⚠️ **Error streaming from Ollama:** {str(e)}"

SYSTEM_PROMPT = """You are an enterprise assistant providing clear, short, well-structured information.
Format your responses with:
- Clear headings (## or ###)
- Numbered lists or bullet points
- Bold text for **key terms**
NO ASCII ART. Use clean text formatting ONLY.
If information is not in context, say: 'This information is not available in the document.'"""


# ----------- GRID OVERLAY HELPER -----------

# Grid configuration
GRID_COLS = 4
GRID_ROWS = 3

def _create_grid_overlay(image_path):
    """
    Create a copy of the image with a labeled grid overlay.
    The grid divides the image into GRID_COLS x GRID_ROWS cells labeled A1-C4.
    
    Returns:
        tuple: (base64_gridded_image, original_img, cell_map)
        cell_map is a dict mapping cell labels like 'A1' to (x1, y1, x2, y2) pixel bounds.
    """
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size

    # Make a copy for the grid overlay
    gridded = img.copy()
    draw = ImageDraw.Draw(gridded)

    cell_w = img_w // GRID_COLS
    cell_h = img_h // GRID_ROWS

    cell_map = {}
    row_labels = [chr(ord('A') + r) for r in range(GRID_ROWS)]  # A, B, C

    # Draw grid lines and labels
    grid_color = (0, 255, 0)   # Green lines for grid (visible on most images)
    label_color = (0, 255, 0)

    try:
        font = ImageFont.truetype("arial.ttf", max(16, min(img_w, img_h) // 25))
    except Exception:
        font = ImageFont.load_default()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w
            y2 = (r + 1) * cell_h

            label = f"{row_labels[r]}{c + 1}"
            cell_map[label] = (x1, y1, x2, y2)

            # Draw cell border
            draw.rectangle([(x1, y1), (x2, y2)], outline=grid_color, width=2)

            # Draw label with a small dark background for readability
            text_x = x1 + 4
            text_y = y1 + 2
            draw.rectangle([(text_x - 2, text_y - 2), (text_x + 30, text_y + 18)], fill=(0, 0, 0))
            draw.text((text_x, text_y), label, fill=label_color, font=font)

    # Convert gridded image to base64
    buffered = BytesIO()
    gridded.save(buffered, format="PNG")
    gridded_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return gridded_b64, img, cell_map


def _highlight_cells(original_img, cell_map, cell_labels, label_text="AI Highlighted Regions"):
    """
    Draw semi-transparent orange overlays on the specified grid cells.
    
    Args:
        original_img: PIL Image (RGB)
        cell_map: dict mapping cell label -> (x1, y1, x2, y2)
        cell_labels: list of cell labels to highlight, e.g. ['A1', 'B3']
        label_text: text to draw at the top
    
    Returns:
        base64 encoded PNG string (with data URI prefix)
    """
    highlighted = original_img.copy()
    overlay = Image.new('RGBA', highlighted.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    orange_fill = (255, 165, 0, 70)     # Semi-transparent orange
    orange_border = (255, 165, 0, 255)  # Solid orange

    highlight_count = 0
    for label in cell_labels:
        label_upper = label.upper().strip()
        if label_upper in cell_map:
            x1, y1, x2, y2 = cell_map[label_upper]
            # Semi-transparent fill
            overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=orange_fill)
            # Solid border
            overlay_draw.rectangle([(x1, y1), (x2, y2)], outline=orange_border, width=3)
            highlight_count += 1

    # Composite the overlay onto the highlighted image
    highlighted = highlighted.convert('RGBA')
    highlighted = Image.alpha_composite(highlighted, overlay)
    highlighted = highlighted.convert('RGB')

    # Add label at top
    draw = ImageDraw.Draw(highlighted)
    try:
        font = ImageFont.load_default()
        draw.rectangle([(0, 0), (highlighted.width, 22)], fill=(0, 0, 0))
        draw.text((10, 4), f"🎯 {label_text} ({highlight_count} regions)", fill=(255, 165, 0), font=font)
    except Exception:
        pass

    buffered = BytesIO()
    highlighted.save(buffered, format="PNG")
    highlighted_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{highlighted_b64}", highlight_count


# ----------- INTERPRET IMAGES WITH VISION MODEL -----------
def interpret_image_with_vision(image_path, question):
    """
    Use LLaVA vision model to interpret and describe what the image shows.
    
    Args:
        image_path: Path to the image file
        question: The user's original question (for context)
    
    Returns:
        A text description of what the image shows
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a technical diagram analyst. Describe ONLY what you can actually see in this image. Do not guess or speculate.

User's question: {question}

Respond in this structure:

## What This Diagram Shows
One clear sentence describing the image.

## Visible Components
- List only components you can clearly identify in the image
- For each, state what it appears to be and where it is in the image (left, center, right, top, bottom)

## Labels and Text
List any text, labels, numbers, or annotations visible in the image exactly as written.

## Answer to the Question
Based on what is visible, answer the user's question. If the image does not contain enough information, say so.

Be concise and factual. NO speculation. NO ASCII ART.""",
                    "images": [image_data]
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        print(f"Error interpreting image {image_path}: {e}")
        return f"Could not interpret image. Error: {str(e)}"


def highlight_diagram_elements(image_path, question):
    """
    Use LLaVA to identify key elements in the diagram using a grid-overlay approach,
    then highlight the relevant grid cells with ORANGE overlays.
    
    The image is divided into a labeled grid (A1-C4). LLaVA identifies which cells
    contain important components, and those cells are highlighted on the original image.
    
    Args:
        image_path: Path to the image file
        question: The user's question to identify relevant components
    
    Returns:
        dict with interpretation and base64 encoded highlighted image
    """
    try:
        # Step 1: Create a gridded version of the image
        gridded_b64, original_img, cell_map = _create_grid_overlay(image_path)
        img_width, img_height = original_img.size

        # Step 2: Send gridded image to LLaVA — ask which cells contain components
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": f"""This image has a green grid overlay with labeled cells (A1-A4 top row, B1-B4 middle row, C1-C4 bottom row).

User's question: {question}

TASK 1 — REASONING:
Explain step-by-step what specific visual shapes or objects you need to look for to answer the user's question. Then describe exactly where you see those objects in the image.

TASK 2 — CELL EXTRACTION:
Based on your reasoning, list the specific grid cells that contain the exact object(s) the user asked about. 
You MUST format each cell exactly like this, including the brackets:
[A1]
[B3]

TASK 3 — FINAL ANSWER:
Answer the user's question clearly based ONLY on what you highlighted.

Be factual. NO ASCII ART.""",
                    "images": [gridded_b64]
                }
            ],
            options={"temperature": 0.1}
        )

        interpretation = response['message']['content']

        # Step 3: Parse which cells LLaVA identified
        import re
        cell_pattern = r'\[([A-Ca-c][1-4])\]'
        found_cells = re.findall(cell_pattern, interpretation)

        # Deduplicate
        found_cells = list(dict.fromkeys([c.upper() for c in found_cells]))

        # Step 4: Highlight those cells on the ORIGINAL image (no grid lines)
        if found_cells:
            highlighted_b64, highlight_count = _highlight_cells(
                original_img, cell_map, found_cells,
                label_text="Orange highlights mark key parts identified by AI"
            )
        else:
            # Fallback: no cells parsed — just add orange border
            fallback = original_img.copy()
            draw = ImageDraw.Draw(fallback)
            draw.rectangle(
                [(4, 4), (img_width - 4, img_height - 4)],
                outline=(255, 165, 0), width=4
            )
            buffered = BytesIO()
            fallback.save(buffered, format="PNG")
            highlighted_b64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            highlight_count = 0

        # Step 5: Clean up interpretation — remove CELL: lines for cleaner display
        clean_interpretation = re.sub(r'CELL:\s*[A-Ca-c]\d\s*-?\s*', '- ', interpretation)

        return {
            "status": "success",
            "interpretation": clean_interpretation,
            "original_image": image_path,
            "highlighted_image": highlighted_b64,
            "analysis_info": {
                "model_used": "llava",
                "image_size": f"{img_width}x{img_height}",
                "annotation_applied": True,
                "highlight_color": "ORANGE",
                "components_highlighted": highlight_count,
                "cells_identified": found_cells
            }
        }

    except Exception as e:
        print(f"Error highlighting diagram {image_path}: {e}")
        return {
            "status": "error",
            "message": f"Could not highlight diagram. Error: {str(e)}"
        }


def analyze_blueprint_component(image_path, component_name):
    """
    Deep analysis of a specific component in a blueprint/schematic.
    
    Args:
        image_path: Path to the blueprint/diagram
        component_name: Name of the component to focus on
    
    Returns:
        Detailed analysis of the component
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": f"""Look at this technical diagram and focus specifically on: '{component_name}'.

IMPORTANT: Only describe what you can actually see. If the component is not visible, say so.

## Is '{component_name}' Visible?
State yes or no. If yes, describe its location in the image.

## What You Can See
Describe the component as it appears — shape, color, size, labels visible on or near it.

## Connections
What other visible parts connect to it?

## Technical Details
Any specifications, numbers, or text labels near this component.

## Function
Based on what you see, explain what this component likely does.

Be factual and concise. NO ASCII ART.""",
                    "images": [image_data]
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        print(f"Error analyzing component {image_path}: {e}")
        return f"Could not analyze component. Error: {str(e)}"


def locate_and_highlight_part(image_path, part_name):
    """
    Locate a specific part in an image using the grid-overlay approach
    and highlight it with ORANGE.
    
    Args:
        image_path: Path to the image file
        part_name: Name of the part to locate
    
    Returns:
        dict with analysis and highlighted image
    """
    try:
        # Step 1: Create gridded image
        gridded_b64, original_img, cell_map = _create_grid_overlay(image_path)
        img_width, img_height = original_img.size

        # Step 2: Ask LLaVA to locate the part using grid cells
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": f"""This image has a green grid overlay with labeled cells (A1-A4 top row, B1-B4 middle row, C1-C4 bottom row).

TASK: Find the '{part_name}' in this image.

TASK 1 — REASONING:
Explain where the '{part_name}' would typically be mathematically or practically located in a real-world scenario. Then look at the image grids to verify if you can actually see it there.

TASK 2 — CELL EXTRACTION:
If you can see the '{part_name}', list the specific grid cells that contain it. 
You MUST format each cell exactly like this, including the brackets:
[A2]

If you CANNOT find '{part_name}', say:
NOT_FOUND

TASK 3 — DESCRIPTION:
Why is this component important? What does it look like?

Be factual. NO ASCII ART.""",
                    "images": [gridded_b64]
                }
            ],
            options={"temperature": 0.1}
        )

        response_text = response['message']['content']

        # Step 3: Parse which cell(s) contain the part
        import re
        cell_pattern = r'\[([A-Ca-c][1-4])\]'
        cell_matches = re.findall(cell_pattern, response_text)
        
        found_cells = list(dict.fromkeys([c.upper() for c in cell_matches]))
        location_found = len(found_cells) > 0

        # Step 4: Highlight on original image
        if found_cells:
            highlighted_b64, highlight_count = _highlight_cells(
                original_img, cell_map, found_cells,
                label_text=f"Found: {part_name.upper()}"
            )
        else:
            # No cells found — draw border as fallback
            fallback = original_img.copy()
            draw = ImageDraw.Draw(fallback)
            draw.rectangle(
                [(4, 4), (img_width - 4, img_height - 4)],
                outline=(255, 165, 0), width=4
            )
            try:
                font = ImageFont.load_default()
                draw.text((15, 15), f"Could not locate: {part_name}", fill=(255, 165, 0), font=font)
            except Exception:
                pass
            buffered = BytesIO()
            fallback.save(buffered, format="PNG")
            highlighted_b64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            highlight_count = 0

        # Clean up response — remove FOUND_IN / CELL lines
        clean_response = re.sub(r'(?:FOUND_IN|CELL):\s*[A-Ca-c]\d\s*-?\s*', '', response_text).strip()

        return {
            "status": "success",
            "part_searched": part_name,
            "location_found": location_found,
            "analysis": clean_response,
            "highlighted_image": highlighted_b64,
            "highlight_info": {
                "color": "ORANGE",
                "style": "Grid cell overlay",
                "cells_highlighted": found_cells,
                "image_size": f"{img_width}x{img_height}"
            }
        }

    except Exception as e:
        print(f"Error locating part in image: {e}")
        return {
            "status": "error",
            "message": f"Could not locate part. Error: {str(e)}"
        }


# ----------- MAIN PROGRAM -----------
if __name__ == "__main__":
    pdf_path = "../documents/leave_policy_dummy.pdf"

    print("Reading PDF...")
    text = load_pdf_text(pdf_path)

    print("Splitting text...")
    chunks = split_text(text)

    print("Creating FAISS index...")
    index = create_index(chunks)

    print("\nSystem Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        question = input("Ask: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question, index, chunks)

        print("\nAnswer:\n", answer)
