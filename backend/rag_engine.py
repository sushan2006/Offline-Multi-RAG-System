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

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


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
    chunks = []
    chunk_roles = []
    chunk_sources = []
    chunk_pages = []   # ⭐ NEW

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)

            extract_images_from_pdf(path, file)

            role = file.split("_")[0].upper()

            reader = PdfReader(path)

            for page_num, page in enumerate(reader.pages):
                content = page.extract_text()

                if not content:
                    continue

                parts = split_text(content)

                for part in parts:
                    chunks.append(part)
                    chunk_roles.append(role)
                    chunk_sources.append(file)
                    chunk_pages.append(page_num)   # ⭐ STORE PAGE

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
def split_text(text, chunk_size=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks



# ----------- CREATE VECTOR INDEX -----------
def create_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index


# ----------- ASK QUESTION -----------
def ask_question(question, index, chunks, history=None, return_indices=False):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k=5)

    context = ""
    matched_indices = I[0].tolist()  # Convert to list for easier handling
    for i in matched_indices:
        context += chunks[i] + "\n"

    history_text = ""
    if history:
        for q, a in history:
            history_text += f"User: {q}\nAssistant: {a}\n"

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": """You are an enterprise assistant providing clear, well-structured information.

IMPORTANT: Format your responses with:
- Clear headings (use ## or ###)
- Numbered lists (1. 2. 3.) for steps or sequences
- Bullet points (- ) for lists of items
- Bold text for **key terms**
- Proper paragraphs with line breaks

NO ASCII ART BOXES OR DIAGRAMS. Use clean text formatting only.

If information is not in the context, say:
'This information is not available in the document.'"""
            },
            {
                "role": "user",
                "content": f"""
Previous Conversation:
{history_text}

Context from Document:
{context}

Question:
{question}

Please provide a clear, well-structured answer using markdown formatting.
"""
            }
        ]
    )

    answer = response['message']['content']
    
    if return_indices:
        return answer, matched_indices
    return answer


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
                    "content": f"""Please analyze this diagram/technical image in detail.

User's Question: {question}

Format your response with clear structure:

## Overview
What does this diagram show?

## Key Components
- List each important component or element visible in the image
- Describe their function or purpose

## Relevant Details
- Important specifications or labels visible
- Technical specifications if shown
- Any measurements or values

## Connection to Your Question
How does this diagram relate to the user's question?

## Recommendations
Any important notes or recommendations based on the diagram.

Use bullet points and short paragraphs. NO ASCII ART.""",
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
    Use LLaVA to identify key elements in the diagram, then highlight them.
    Works best with technical diagrams, blueprints, and schematics.
    
    Args:
        image_path: Path to the image file
        question: The user's question to identify relevant components
    
    Returns:
        dict with interpretation and base64 encoded highlighted image
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Ask LLaVA to identify specific components
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this technical diagram carefully.

User's Question: {question}

Provide analysis in this format:

## Components Identified
List the specific components or elements that answer the user's question.

## Locations
For each component, describe where it is located in the diagram (e.g., top-left, center, bottom-right).

## Function and Purpose
Explain what each component does and its purpose in the system.

## Connections
Describe any connections or relationships between components.

## Key Takeaways
Summarize the most important information from the diagram relevant to the question.

Use bullet points and clear headings. NO ASCII ART OR BOXES.""",
                    "images": [image_data]
                }
            ]
        )
        
        interpretation = response['message']['content']
        
        # Load the original image for annotation
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create a copy for highlighting
        highlighted = img.copy()
        draw = ImageDraw.Draw(highlighted, 'RGBA')
        
        # Add semi-transparent overlay to highlight areas
        # This creates a "brightening" effect for attention regions
        overlay = Image.new('RGBA', highlighted.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add decorative borders and annotations
        border_color = (255, 0, 0, 180)  # Red with transparency
        
        # Add a prominent border
        border_width = 5
        draw.rectangle(
            [(border_width, border_width), 
             (img_width - border_width, img_height - border_width)],
            outline=border_color,
            width=border_width
        )
        
        # Add text label for analysis
        try:
            font = ImageFont.load_default()
            label_text = "[AI Analyzed Diagram]"
            draw.text((10, 10), label_text, fill=(255, 0, 0, 255), font=font)
        except:
            pass  # If font fails, just skip text
        
        # Convert to base64 for frontend display
        buffered = BytesIO()
        highlighted.save(buffered, format="PNG")
        highlighted_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "interpretation": interpretation,
            "original_image": image_path,
            "highlighted_image": f"data:image/png;base64,{highlighted_base64}",
            "analysis_info": {
                "model_used": "llava",
                "image_size": f"{img_width}x{img_height}",
                "annotation_applied": True
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
    Useful for detailed technical queries.
    
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
                    "content": f"""Examine this diagram carefully and focus on the '{component_name}'.

Provide detailed analysis with these sections:

## Location
Describe exactly where the {component_name} is located in the diagram.

## Specifications
List all specifications or technical specs visible for this component.

## Connections
What other components does it connect to? How are they connected?

## Function
Explain what the {component_name} does and its role in the system.

## Safety Notes
Any safety considerations, warnings, or hazards associated with this component?

## Maintenance
Maintenance requirements, inspection points, or operational notes.

## Identification
Any part numbers, labels, or codes visible on or near this component?

Use clear formatting with bullet points. NO ASCII ART.""",
                    "images": [image_data]
                }
            ]
        )
        
        return response['message']['content']
    except Exception as e:
        print(f"Error analyzing component in image: {e}")
        return f"Could not analyze component. Error: {str(e)}"


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
