# AI-Powered Image Generator — Talrn ML Internship Task

A fully local **text-to-image generator** built using the open-source model  
**Stable Diffusion v1.5** (via Hugging Face Diffusers) and a simple **Streamlit web UI**.

This project meets all the requirements:
- Open-source model only  
- Local execution (CPU + GPU optional)  
- Text prompt input + adjustable generation settings  
- Style presets, negative prompts & prompt enhancements  
- Real-time progress + ETA  
- Image downloads (PNG + JPEG)  
- Watermarking for ethical AI use  
- Metadata storage  
- Complete documentation (README + hardware guide + ethics file)

---

# 🚀 Features

### ✔ Text-to-image generation  
Enter a prompt (e.g., *“a futuristic Indian city at sunset”*) and generate 1–4 images.

### ✔ Prompt engineering built-in  
The app auto-adds high-quality modifiers:
- *ultra detailed, 4k, cinematic, high clarity*

### ✔ Style presets  
Choose from:
- Default  
- Photorealistic  
- Artistic  
- Cartoon  
- Cyberpunk  

### ✔ Negative prompts  
Remove unwanted elements like:
- *blurry, text, watermark, distorted anatomy, low quality*

### ✔ Adjustable generation settings  
- Width / Height  
- Steps (8–40)  
- Guidance scale  
- Seed (optional)

### ✔ Progress bar & ETA  
Real-time callback integration with `diffusers` for step-by-step progress.

### ✔ Multi-format image download  
- PNG  
- JPEG

### ✔ Watermarking  
All images include a small, transparent:
```
AI-generated (Talrn)
```

### ✔ Metadata storage  
Each generation saves:
- prompt  
- negative prompt  
- width/height  
- steps  
- guidance scale  
- timestamp  
- image paths  

Saved under:
```
outputs/<timestamp>/
```

---

# 📁 Project Structure

```
Talrn-image-generator/
│
├── streamlit_app.py          # Main Streamlit app
├── generate_cli.py           # CLI generator (optional)
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── ETHICAL.md                # Ethical usage guidelines
├── hardware_notes.md         # CPU/GPU installation guide
├── LICENSE                   # MIT license
│
└── outputs/
      └── sample_outputs/     # Example images
```

---

# 📥 Installation

## 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
```

Activate it (Windows):
```bash
.venv\Scripts\activate
```

## 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

## 3️⃣ Run the App
```bash
streamlit run streamlit_app.py
```

---

# 🔧 Model Details

This project uses the open-source model:

**runwayml/stable-diffusion-v1-5**  
https://huggingface.co/runwayml/stable-diffusion-v1-5

### First-time setup (Hugging Face)
Accept the model license:
- Login / Signup → go to model page  
- Click **“I Accept”**  

(Optional) Login from terminal for faster downloads:
```bash
huggingface-cli login
```

---

# 💻 Hardware Requirements

See detailed guide in **hardware_notes.md**

### CPU
- Works everywhere  
- Recommended: 16GB RAM  
- Image takes ~20–90 seconds on CPU

### GPU (optional)
- NVIDIA CUDA recommended  
- Huge speed boost (10×–20× faster)  
- Works with RTX GPUs (2060, 3060, 4060, etc.)

---

# 🧱 Technology Stack

- **Python 3.10+**  
- **PyTorch** (CPU or GPU)
- **Hugging Face Diffusers**
- **Transformers**
- **Streamlit** (web UI)
- **Pillow**, **safetensors**, **numpy**, **tqdm**

---

# 📑 Prompt Engineering Tips

- Use **comma-separated visual concepts**:
  ```
  subject, environment, camera style, lighting, quality
  ```
- Add 3–5 high-quality boosters:
  - *ultra detailed*
  - *cinematic*
  - *sharp focus*
  - *4k*
- Use **negative prompts** to clean output.
- Increase guidance (7–9) for stricter prompt following.
- Increase steps (20–30) for crisp results.

---

# ⚠️ Limitations

- CPU inference is slow  
- SD v1.5 struggles with complex scenes containing 3+ subjects  
- High memory usage (3GB+)  
- Not suitable for very large images on CPU  

---

# 🔮 Future Improvements

- Add SDXL 1.0 model support  
- Add Real-ESRGAN upscaler  
- Add LoRA fine-tuning support  
- Add prompt templates  
- Add login + user history  
- Support async background workers  

---

# 📌 Ethical Usage

See `ETHICAL.md`.  
Watermarking is applied automatically to ensure transparency.

---
