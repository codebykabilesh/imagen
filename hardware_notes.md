# Hardware Notes — CPU & GPU Installation Guide

This project supports both **CPU-only** systems and **GPU-accelerated** setups.  
Stable Diffusion runs on many hardware configurations, but performance depends on CPU, GPU, RAM, and VRAM.

Below are the complete setup instructions.

---

# 1. CPU Installation (Recommended for Testing)

Most users without a GPU can run Stable Diffusion on CPU.  
It is slower, but highly compatible.

### Install CPU PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install Remaining Dependencies
```bash
pip install diffusers transformers accelerate safetensors streamlit pillow tqdm numpy
```

### CPU Performance Tips
Expect **20–90 seconds** per image depending on resolution.

Recommended settings:

- **Width/Height:** 256–384 px  
- **Steps:** 10–25  
- **Guidance scale:** 6–9  

Avoid:

- 512×512 resolution  
- 30+ steps on weak CPUs  

The application already uses `low_cpu_mem_usage=True` for stability.

---

# 2. GPU Installation (NVIDIA CUDA Preferred)

If you have an NVIDIA GPU, generation speed improves by **10x–30x**.

### Step 1 — Install CUDA-Compatible PyTorch
Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Find other versions here:  
https://pytorch.org/get-started/locally/

### Step 2 — Install Remaining Dependencies
```bash
pip install diffusers[torch] transformers accelerate safetensors streamlit pillow tqdm numpy
```

### Step 3 — Enable GPU Mode in Code

In `streamlit_app.py`, inside `load_pipeline()`:

Replace:
```python
pipe = pipe.to("cpu")
```

With:
```python
pipe = pipe.to("cuda")
```

### Optional — Enable FP16 for Faster Generation
Use FP16 if GPU VRAM ≥ 6GB:
```python
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")
```

---

# 3. Low-VRAM GPUs (2GB–4GB) — Optimization Tips

If your GPU is older or small (GTX 1650, MX series):

- Use **256–384 px** resolution  
- Steps: **10–20**  
- Guidance: **6–8**  
- Enable `low_cpu_mem_usage=True`  
- Use `torch.float16` when possible  
- Close Chrome, games, and other GPU apps  

---

# 4. Windows Virtual Memory (Pagefile) — IMPORTANT

If you see errors like:

```
OSError: The paging file is too small
MemoryError
```

Fix it:

1. Open  
   **Control Panel → System → Advanced system settings → Performance → Settings**
2. Go to **Virtual Memory**
3. Set:
   - Initial size: **16000 MB**
   - Maximum size: **32000 MB**
4. Restart your computer.

---

# 5. Common Errors & Solutions

### ❌ CUDA out of memory
Fix:
- Reduce resolution  
- Reduce steps  
- Lower guidance scale  
- Use FP16  
- Close GPU-heavy apps  

### ❌ Model too large / MemoryError
Fix:
- Increase pagefile (Windows)  
- Use 256/384 px  
- Use SD 1.5 instead of SDXL  

### ❌ Slow generation
Fix:
- Keep resolution low  
- Steps ≤ 20  
- Close unnecessary programs  

---

# 6. Recommended Hardware

### Minimum (works on CPU)
- Intel i5 / Ryzen 5  
- 8GB RAM  
- No GPU needed  

### Good Setup
- Intel i7 / Ryzen 7  
- 16GB–32GB RAM  
- NVIDIA RTX 2060 / 3060  
- 6GB–12GB VRAM  

### Best Setup
- RTX 3080 / 4080 / 4090  
- 16GB VRAM+  
- Very fast encoding

---

# 7. Cloud GPU Options (Optional)

If you want faster generation:

- Google Colab (Free GPU)  
- Kaggle Notebooks  
- RunPod  
- Vast.ai  
- Paperspace  

You can upload your repo and run the Streamlit app remotely.

---

# 8. Summary

- CPU mode works everywhere but is slower.  
- GPU mode is optional but gives massive speed improvements.  
- This project includes both CPU and GPU instructions to meet the internship requirement.  
- Make sure to adjust pagefile / VRAM settings to prevent errors.  

---
