# 🎮 GPU Setup for RTX 5050 (NVIDIA CUDA)

> **⚠️ CRITICAL:** If you install PyTorch via normal `pip install -r requirements.txt`, you will get **CPU-only** PyTorch. Follow these steps to use your RTX 5050.

---

## Step 1: Check Your CUDA Version

Open PowerShell or Command Prompt and run:

```powershell
nvidia-smi
```

Look at the top-right for **CUDA Version** (e.g. 12.4, 12.6). Use that to pick the install command below.

---

## Step 2: Install PyTorch with CUDA **FIRST**

**Install PyTorch with CUDA BEFORE any other packages.** This prevents pip from pulling in CPU-only PyTorch.

### CUDA 12.1
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 12.4
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### CUDA 12.6
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CUDA 11.8 (older drivers)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

👉 **Not sure?** Go to [pytorch.org](https://pytorch.org) → Select **Windows**, **Pip**, and your CUDA version → Copy the command.

---

## Step 3: Install the Rest of the Project

After PyTorch CUDA is installed:

```powershell
pip install -r requirements.txt
```

`sentence-transformers` will use the already-installed PyTorch (with CUDA) and won't overwrite it.

---

## Step 4: Verify GPU in Python

Run the verification script:

```powershell
python verify_gpu.py
```

Or manually:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
```

Expected output:
```
CUDA available: True
Device name: NVIDIA GeForce RTX 5050
```

---

## Step 5: Confirm the RAG Model Uses GPU

The project already loads the embedding model on GPU in `backend/rag_engine.py`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
```

When you start the backend, you should see:
```
Loading embedding model on cuda...
```

If you see `Loading embedding model on cpu...` → PyTorch CUDA was not installed correctly.

---

## Ollama (LLaVA / Qwen) — Main CPU Hog

Ollama runs the text model (qwen2.5) and vision model (llava). **If GPU is at 4% and CPU at 83%, Ollama is likely using CPU.**

### Check where Ollama loads models

```powershell
ollama ps
```

Look at the **Processor** column:
- `100% GPU` = using GPU
- `100% CPU` = using CPU (this causes high CPU, low GPU)

### Fix Ollama to use GPU

1. **Update Ollama** — Get the latest from [ollama.com](https://ollama.com)
2. **Update NVIDIA drivers** — RTX 5050 needs driver 531+
3. **Restart Ollama** — Quit from system tray, start again
4. **Pre-load models on GPU** — Run once before using the app:
   ```powershell
   ollama run qwen2.5:0.5b
   ollama run llava
   ```
   Then `ollama ps` — both should show `100% GPU`
5. **Check env vars** — Ensure `CUDA_VISIBLE_DEVICES` is NOT set to `-1` (that forces CPU)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch.cuda.is_available()` is False | Reinstall PyTorch with the correct `--index-url` for your CUDA version. Uninstall first: `pip uninstall torch torchvision torchaudio` |
| Wrong CUDA version | Check `nvidia-smi` and use the matching cu1xx URL |
| "No module named torch" | Run Step 2 first, then Step 3 |
| Model still on CPU | Ensure you ran Step 2 **before** `pip install -r requirements.txt` |
| **`ImportError: cannot import name 'InterpolationMode' from 'torchvision.transforms'`** | Your torchvision is too old. Upgrade PyTorch stack: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade` (use cu121/cu126 if that matches your `nvidia-smi`) |
| **GPU at 4%, CPU at 83%** | Ollama is using CPU. Run `ollama ps` to confirm. Update Ollama + drivers, restart, pre-load models with `ollama run llava`. See Ollama section above. |
| **FAISS GPU not working** | Install: `pip install faiss-gpu`. If it fails (e.g. on some Windows setups), the app falls back to FAISS CPU automatically. |
