# Ollama Image Analysis Setup Guide

## What You Now Have

Your RAG system now supports **multi-modal blueprint/diagram analysis** using the **LLaVA** vision model from Ollama.

### NEW: Automatic Image Analysis 🚀

**You no longer need to specify image names!** When the system shows you an image and you ask "analyze the above image", it automatically:
1. Remembers which image was shown
2. Detects that you're asking about an image
3. Uses LLaVA to analyze it
4. Returns the analysis with highlighted results

### Installed Ollama Models

**Recommended:** `llava` (already in your code)

Other options:
- `bakllava` - Optimized version, better detail recognition
- `minicpm-v` - Lighter weight, excellent for precise visual understanding

## Installation Steps

### 1. Install Dependencies
Run this in your workspace:
```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Image processing
- `Pillow` - Image manipulation
- Plus all existing dependencies

### 2. Download Ollama Model

If you haven't already, download the LLaVA model:
```bash
ollama pull llava
```

Or download the optimized version:
```bash
ollama pull bakllava
```

### 3. Start Ollama Service
```bash
ollama serve
```

Keep this running in the background while your FastAPI server runs.

## How to Use: Automatic Image Analysis ⚡

### Scenario 1: Simple Image Analysis
```
1. User asks: "Tell me about cargo handling procedures"
2. System returns: Text answer + shows relevant ship diagram
3. User asks: "Analyze the above image"
4. System automatically: Detects image request, highlights diagram, provides visual analysis
```

### Scenario 2: Fluent Conversation
```
1. User: "What are the main components of the propulsion system?"
2. System: Returns explanation + shows engine diagram
3. User: "Explain what's shown in the diagram"
4. System: Automatically analyzes the image LLaVA showed, no image filename needed!
5. User: "What about the oil pump specifically?"
6. System: Can analyze the same image for different components
```

### Supported Query Patterns
The system automatically detects when you want image analysis with these phrases:
- "Analyze the above image"
- "What is shown in the image?"
- "Explain the diagram"
- "Describe the above blueprint"
- "Show me where is [component]"
- "Highlight the [component]"
- "Tell me about the above image"
- "Interpret this diagram"
- And many more natural language variations!

## API Endpoints

### The `/ask` Endpoint (Enhanced)
Now works with BOTH text and image queries automatically!

**Example 1: Regular Text Query**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the navigation rules?",
    "role": "CAPTAIN"
  }'
```

Response includes relevant images and their interpretations:
```json
{
  "answer": "Navigation rules...",
  "source": ["NAV_navigation_rules.pdf"],
  "images": ["NAV_navigation_rules.pdf_page0_0.jpx"],
  "image_details": [{
    "image": "NAV_navigation_rules.pdf_page0_0.jpx",
    "interpretation": "This diagram shows..."
  }],
  "analysis_type": "document_query"
}
```

**Example 2: Automatic Image Analysis**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze the above image",
    "role": "CAPTAIN"
  }'
```

Response with highlighted and analyzed image:
```json
{
  "answer": "The ship diagram shows the cargo holds positioned...",
  "source": ["NAV_navigation_rules.pdf"],
  "images": ["NAV_navigation_rules.pdf_page0_0.jpx"],
  "image_details": [{
    "image": "NAV_navigation_rules.pdf_page0_0.jpx",
    "interpretation": "Detailed LLaVA analysis...",
    "highlighted_image": "data:image/png;base64,...",
    "analysis_info": {
      "model_used": "llava",
      "image_size": "1024x768",
      "annotation_applied": true
    }
  }],
  "analysis_type": "image_analysis"
}
```

## Legacy Endpoints (Still Available)

If you need to directly specify image names, these endpoints are still available:

### 1. **Highlight Diagram Elements**
Analyzes a diagram and highlights key components with interpretation.

**Endpoint:** `POST /analyze-diagram`

**Request:**
```json
{
  "image_name": "ENGINE_engine_operations.pdf_page0_0.jpx",
  "question": "Where is the oil filter valve and how do I maintain it?"
}
```

**Response:**
```json
{
  "status": "success",
  "interpretation": "The oil filter is located at the top-right of the diagram...",
  "highlighted_image": "data:image/png;base64,...",
  "original_image": "extracted_images/ENGINE_engine_operations.pdf_page0_0.jpx",
  "analysis_info": {
    "model_used": "llava",
    "image_size": "1024x768",
    "annotation_applied": true
  }
}
```

### 2. **Analyze Specific Component**
Deep technical analysis of a specific component in a blueprint.

**Endpoint:** `POST /analyze-component`

**Request:**
```json
{
  "image_name": "ENGINE_engine_operations.pdf_page0_0.jpx",
  "component_name": "Oil Pump"
}
```

**Response:**
```json
{
  "component": "Oil Pump",
  "image": "ENGINE_engine_operations.pdf_page0_0.jpx",
  "analysis": "The Oil Pump is a centrifugal pump located at coordinates... It connects to... Safety considerations include..."
}
```

### 3. **Interpret Image (Simple)**
Returns text interpretation without highlighting.

**Endpoint:** `POST /interpret-image-detailed`

**Request:**
```json
{
  "image_name": "ENGINE_engine_operations.pdf_page0_0.jpx",
  "question": "What components are shown in this diagram?"
}
```

**Response:**
```json
{
  "image": "ENGINE_engine_operations.pdf_page0_0.jpx",
  "question": "What components are shown in this diagram?",
  "interpretation": "This is a technical diagram showing..."
}
```

## Model Capabilities

The LLaVA model can:
✅ Identify specific components in technical diagrams  
✅ Read text labels and annotations in images  
✅ Describe spatial relationships between components  
✅ Provide technical specifications visible in diagrams  
✅ Interpret electrical schematics and P&ID diagrams  
✅ Recognize piping connections and flow paths  
✅ Identify safety features and warning labels  
✅ **Automatically respond to image questions without image filenames**

## Frontend Integration

### Simple Example: No Image Name Needed!
```html
<form id="askForm">
  <textarea id="question" placeholder="Ask about the document or image..."></textarea>
  <button type="submit">Ask</button>
</form>

<div id="response">
  <p id="answer"></p>
  <img id="image" src="" alt="">
</div>

<script>
  document.getElementById('askForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = document.getElementById('question').value;
    
    const response = await fetch('/ask', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: question,  // Can be "Analyze the above image" - no filenames!
        role: 'CAPTAIN'
      })
    });
    
    const data = await response.json();
    document.getElementById('answer').textContent = data.answer;
    
    if (data.image_details && data.image_details[0]) {
      const detail = data.image_details[0];
      if (detail.highlighted_image) {
        // Use highlighted version if available
        document.getElementById('image').src = detail.highlighted_image;
      }
    }
  });
</script>
```

### Advanced Example: Track Image Analysis
```html
<script>
  // Response includes 'analysis_type' field
  // "document_query" = normal text search
  // "image_analysis" = automatic image analysis
  
  const response = await fetch('/ask', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      question: userQuestion,
      role: userRole
    })
  });
  
  const data = await response.json();
  
  if (data.analysis_type === 'image_analysis') {
    console.log('System automatically analyzed an image!');
    console.log('Highlighted image:', data.image_details[0].highlighted_image);
  }
</script>
```

## Performance Tips

1. **Automatic image detection** is instant - no extra API calls needed
2. **Larger images** = more detail but slower LLaVA processing
3. **Specific questions** get better results than generic ones
4. **BakLLaVA** is typically 10-20% faster than LLaVA
5. **MiniCPM-V** is the fastest for real-time applications
6. The system keeps track of last 5 shown images per user for fluent conversations

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### "Could not connect to Ollama"
Make sure Ollama service is running:
```bash
ollama serve
```

### "Model not found"
Download the model:
```bash
ollama pull llava
```

### Slow image processing
- Try a smaller model like `bakllava` or `minicpm-v`
- Reduce image size before sending
- Use `/interpret-image-detailed` instead of `/analyze-diagram`

### "System doesn't detect my image question"
Try these phrases instead:
- "Where is the [component]?"
- "Show me [component] in the diagram"
- "Highlight the [component]"
- "Tell me about the diagram"
- "What does the image show?"

### Image analysis returns generic response
- Use more specific questions: "Where is the oil pump and what does it connect to?"
- The system works best with multiple related questions about the same image

## Next Steps

1. ✅ Install dependencies
2. ✅ Pull Ollama model: `ollama pull llava`
3. ✅ Start Ollama: `ollama serve`
4. ✅ Run your FastAPI app: `python -m uvicorn main:app --reload`
5. ✅ Test endpoints with sample images from `extracted_images/`

---

**You're all set!** Your system can now analyze technical diagrams and highlight specific components for engineers.
