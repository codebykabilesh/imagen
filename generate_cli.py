# generate_cli.py (replace top part of your file with this version)
import os
import time
import json
import torch

from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_pipeline():
    # Reduce CPU threads to cut memory overhead during load
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    # Minor env tweak (can help on some setups)
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    print("Loading model with low_cpu_mem_usage=True ... (this may still take a few minutes)")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        safety_checker=None,          # optional: remove if you want the safety checker enabled
        use_safetensors=True,
        low_cpu_mem_usage=True        # <- very important for low-RAM machines
    )
    # keep on CPU
    pipe = pipe.to("cpu")
    return pipe

def main():
    pipe = load_pipeline()

    prompt = "a futuristic city at sunset, highly detailed, cinematic, 4k"
    width, height = 384, 384   # conservative size for 16GB RAM
    steps = 20
    guidance = 7.5

    print(f"Generating image ({width}x{height}, steps={steps}) ...")
    start = time.time()
    result = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance)
    image = result.images[0]
    elapsed = time.time() - start

    out_path = os.path.join(OUT_DIR, f"test_output_{int(time.time())}.png")
    image.save(out_path)
    print(f"Saved {out_path} (took {elapsed:.1f} seconds)")
    # save metadata
    meta = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "elapsed_sec": elapsed,
        "timestamp": time.time()
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

if __name__ == "__main__":
    main()
