
# Hardware Notes — CPU & GPU Installation Guide

This project supports both **CPU-only** systems and **GPU-accelerated** setups.  
Stable Diffusion can run on almost any machine, but performance varies greatly.

Below are clear setup instructions and performance recommendations for each environment.

---

# 1. CPU Installation (Recommended for Testing)

Most users without a GPU can run Stable Diffusion using CPU mode.  
This is slower but supports a wide range of hardware.

### Install CPU PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Install Remaining Dependencies
bash
Copy code
pip install diffusers transformers accelerate safetensors streamlit pillow tqdm numpy
CPU Performance Tips
Expect 20–90 seconds per image depending on resolution.

Use optimized parameters:

Width/Height: 256–384px

Steps: 10–25

Guidance scale: 6–9

Do NOT use 512×512 or >30 steps on weak CPUs.

The model uses low_cpu_mem_usage=True internally for stability.

2. GPU Installation (NVIDIA CUDA Preferred)
If you have an NVIDIA GPU, you will get 10x–20x faster inference.

Step 1 — Install CUDA-Compatible PyTorch
Example (CUDA 11.8):

bash
Copy code
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
For other CUDA versions, check:
https://pytorch.org/get-started/locally/

Step 2 — Install Diffusers + Extras
bash
Copy code
pip install diffusers[torch] transformers accelerate safetensors streamlit pillow tqdm
Step 3 — Enable GPU Mode in Code
Inside load_pipeline() in streamlit_app.py:

Replace:

python
Copy code
pipe = pipe.to("cpu")
With:

python
Copy code
pipe = pipe.to("cuda")
Optional (Recommended)
Enable half-precision for faster performance:

python
Copy code
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")
3. Low-VRAM GPU (2GB–4GB) Tips
If your GPU is weak (e.g., GTX 1650, MX series, older cards):

Reduce image size → 256 or 384

Reduce steps → 10–20

Use:

python
Copy code
low_cpu_mem_usage=True
Avoid high guidance > 10

Use FP16 (torch.float16) when possible

Close other GPU apps (Chrome, games, IDE tools)

4. Windows Pagefile / Virtual Memory
Stable Diffusion sometimes needs more RAM and virtual memory.

If you see:

vbnet
Copy code
OSError: The paging file is too small
MemoryError
Fix:

Open:
Control Panel → System → Advanced → Performance → Settings

Go to Virtual Memory

Set:

Initial size: 16000 MB

Maximum size: 32000 MB

Click OK and restart.

5. Common Issues & Fixes
Issue: “CUDA out of memory”
Fix:

Reduce resolution

Reduce steps

Reduce guidance

Enable FP16

Restart programs using GPU

Issue: “Model too large to load”
Fix:

Increase pagefile (Windows)

Close other memory-heavy applications

Use smaller checkpoints (SD 1.5 instead of SDXL)

Issue: Slow generation (CPU)
Fix:

Set width/height to ~256–384

Steps 10–20

Use shared compiler optimizations in PyTorch

Increase thread count if your CPU supports it

6. Recommended Hardware
Minimum:
CPU: Intel i5 / Ryzen 5

RAM: 8GB

No GPU required

Optimal:
CPU: Intel i7 / Ryzen 7

RAM: 16GB–32GB

GPU: NVIDIA RTX 3060 / 4060 or better

VRAM: 6GB+

High-end:
RTX 3080 / 4080 / 4090

Fastest inference, supports larger models (SDXL)

7. Cloud GPU Options (Optional)
If you want faster/cheaper GPU compute:

Google Colab (Free + Pro)

Kaggle Notebooks (free GPU quota)

RunPod.io

Vast.ai

Paperspace

You can upload your repository and run the Streamlit app remotely with streamer tunnels.

8. Summary
CPU mode works everywhere but is slow → suitable for Talrn task.

GPU mode is optional and provides huge speedups.

This project includes both CPU and GPU instructions to meet the internship requirement.

Pagefile adjustments may be required on Windows.

Recommended to keep image size small when using CPU.



