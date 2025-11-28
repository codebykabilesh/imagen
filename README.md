# AI-Powered Image Generator - Talrn

## Overview
A local text → image generator built with Hugging Face `diffusers` (Stable Diffusion v1.5) and Streamlit. 
Features: prompt entry, style presets, prompt engineering, negative prompts, progress & ETA, watermarking, metadata storage, multi-format downloads (PNG/JPEG).

## Quick demo
1. Create venv, install requirements (see below).
2. Run: `streamlit run streamlit_app.py`
3. Open `http://localhost:8501` and enter a prompt.

## Project structure
- `streamlit_app.py` — Streamlit UI (main app)
- `generate_cli.py` — minimal CLI quick-test generator
- `requirements.txt` — pip dependencies
- `outputs/` — generated images + metadata (created automatically)
- `hardware_notes.md` — GPU/CPU installation & tips
- `ETHICAL.md` — ethical usage & content filtering guidance

## Requirements
- Python 3.10 or 3.11
- 16 GB RAM recommended for CPU runs; GPU recommended for speed.

## Install (CPU example)
```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate
pip install 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors streamlit pillow tqdm
