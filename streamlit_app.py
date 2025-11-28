# streamlit_app.py
import os
import time
import json
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline

# ----------------- Config -----------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT_ROOT = "outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

# ----------------- Prompt engineering presets -----------------
QUALITY_DESCRIPTORS = [
    "highly detailed", "ultra-detailed", "8k", "4k", "photorealistic",
    "professional photography", "award-winning", "cinematic lighting",
    "sharp focus", "intricate", "high contrast", "vibrant colors"
]

NEGATIVE_PRESETS = {
    "General (recommended)": "blurry, low quality, lowres, distorted, bad anatomy, extra limbs, watermark, text",
    "Portraits": "deformed face, bad eyes, bad mouth, extra limbs, double face, low quality, watermark",
    "Architecture / City": "weird buildings, malformed structures, overlapping objects, low quality, watermark",
    "Cartoon/Anime": "messy lines, muddy colors, blurry, low detail, watermark",
    "Remove Text Only": "text, watermark"
}

STYLE_PRESETS = {
    "Default": "",
    "Photorealistic": "photorealistic, ultra-detailed, professional photography",
    "Artistic": "highly detailed, artstation, masterpiece",
    "Cartoon": "cartoon, cel-shading, bold colors",
    "Cyberpunk": "neon lights, cyberpunk cityscape, cinematic"
}

# ----------------- Helpers -----------------
def watermark_image(img: Image.Image, text="AI-generated (Talrn)", style="minimal"):
    drawable = ImageDraw.Draw(img)
    font_size = max(12, img.width // 40)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = drawable.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = img.width - w - 12
    y = img.height - h - 8

    if style == "none":
        return img
    if style == "clean":
        drawable.text((x, y), text, fill=(255, 255, 255, 220), font=font)
    elif style == "shadow":
        drawable.text((x + 2, y + 2), text, fill=(0, 0, 0, 180), font=font)
        drawable.text((x, y), text, fill=(255, 255, 255, 230), font=font)
    elif style == "minimal":
        drawable.text((x, y), text, fill=(255, 255, 255, 160), font=font)
    else:
        drawable.text((x, y), text, fill=(255, 255, 255, 200), font=font)
    return img

def save_image_and_meta_formats(img: Image.Image, prompt: str, meta_params: dict, out_dir: str):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    base = f"{ts}_{uid}"
    paths = {}
    # Save PNG (lossless)
    png_path = os.path.join(out_dir, base + ".png")
    img.save(png_path, format="PNG")
    paths["PNG"] = png_path
    # Save JPEG (converted)
    jpg_path = os.path.join(out_dir, base + ".jpg")
    img.convert("RGB").save(jpg_path, format="JPEG", quality=92)
    paths["JPEG"] = jpg_path

    meta = {
        "prompt_user": meta_params.get("user_prompt"),
        "final_prompt": prompt,
        "negative_prompt": meta_params.get("negative_prompt"),
        "params": meta_params,
        "files": paths,
        "timestamp_utc": ts
    }
    meta_file = os.path.join(out_dir, "metadata.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    else:
        data = []
    data.append(meta)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return paths, meta_file

def build_final_prompt(user_prompt: str, style_extra: str, quality_list: list):
    parts = [p.strip() for p in (user_prompt, style_extra) if p and p.strip()]
    quality = ", ".join([q for q in quality_list if q and q.strip()])
    if quality:
        parts.append(quality)
    final = ", ".join([p for p in parts if p])
    return final

# ----------------- Model loading (cached) -----------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    st.info("Loading model — this may take a few minutes the first time.")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        safety_checker=None
    )
    pipe = pipe.to("cpu")
    return pipe

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Text → Image (Local)", layout="centered")
st.title("Text → Image Generator (Local — CPU-ready)")

st.markdown("Enter a prompt, choose settings, then click **Generate**. Progress and ETA will display while the model runs (CPU is slow).")

with st.sidebar:
    st.header("Model / Hardware")
    st.write("Model:", MODEL_ID)
    st.write("Device:", "cpu (streamlit runs on local machine)")
    st.markdown("---")
    st.header("Prompt engineering")
    chosen_style = st.selectbox("Style preset", list(STYLE_PRESETS.keys()), index=0)
    style_extra = STYLE_PRESETS[chosen_style]
    st.write("Quality descriptors (pick helpful ones):")
    quality_defaults = ["highly detailed", "4k", "professional photography"]
    chosen_quality = st.multiselect("Quality descriptors", QUALITY_DESCRIPTORS, default=quality_defaults)
    neg_preset = st.selectbox("Negative prompt preset", list(NEGATIVE_PRESETS.keys()))
    negative_prompt_input = st.text_area("Negative prompt (editable)", value=NEGATIVE_PRESETS[neg_preset], height=80)
    auto_augment = st.checkbox("Auto-augment prompt with quality descriptors", value=True)
    st.markdown("---")
    st.header("Watermark")
    watermark_style = st.selectbox("Watermark style", ["minimal", "clean", "shadow", "none"], index=0)
    st.checkbox("Show watermark on saved images", value=True, key="show_wm")
    st.markdown("---")
    st.write("Tips:")
    st.write("- Use 256–384 for quick tests on CPU.")
    st.write("- Increase steps & resolution for higher quality (takes longer).")

# Main inputs
prompt = st.text_area("Prompt", value="a futuristic city at sunset, highly detailed, cinematic, 4k", height=140)
num_images = st.slider("Number of images", min_value=1, max_value=4, value=1)
width = st.selectbox("Width", [256, 384, 512], index=1)
height = st.selectbox("Height", [256, 384, 512], index=1)
steps = st.slider("Steps", min_value=8, max_value=50, value=20)
guidance = st.slider("Guidance scale", min_value=1.0, max_value=12.0, value=7.5)
seed_input = st.text_input("Seed (leave blank for random)", value="")
negative_prompt = negative_prompt_input  # from sidebar editable box

# load model
pipe = load_pipeline()

# placeholders for progress UI
progress_bar = st.progress(0)
status_text = st.empty()
eta_text = st.empty()
results_container = st.container()

gen_button = st.button("Generate")

if gen_button:
    blocklist = ["porn", "nsfw", "child"]
    if any(b in prompt.lower() for b in blocklist):
        st.error("Prompt contains prohibited content.")
    elif len(prompt.strip()) < 3:
        st.error("Prompt too short.")
    else:
        out_dir = os.path.join(OUT_ROOT, datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)

        # compose final prompt
        if auto_augment:
            final_prompt = build_final_prompt(prompt, style_extra, chosen_quality)
        else:
            final_prompt = prompt.strip()

        # create generator for reproducibility if seed given
        generator = None
        if seed_input.strip():
            try:
                seed_val = int(seed_input.strip())
            except Exception:
                seed_val = int(uuid.uuid4().int % (2**31))
            generator = torch.Generator("cpu").manual_seed(seed_val)
        else:
            seed_val = None

        total_steps = steps
        state = {"start_time": None}

        def diffusers_callback(step: int, timestep: int, latents):
            if state["start_time"] is None:
                state["start_time"] = time.time()
            try:
                raw_percent = ((step + 1) / max(1, total_steps)) * 100
            except Exception:
                raw_percent = 0.0
            percent = int(max(0, min(100, round(raw_percent))))
            try:
                progress_bar.progress(percent)
            except Exception:
                pass
            elapsed = time.time() - state["start_time"]
            completed_steps = max(1, (step + 1))
            avg_per_step = elapsed / completed_steps
            remaining = avg_per_step * max(0, (total_steps - completed_steps))
            eta_text.markdown(f"**Progress:** {percent}% — elapsed: {int(elapsed)}s — ETA: {int(remaining)}s")

        images = []
        with st.spinner("Generating images... (this may take time on CPU)"):
            for img_i in range(num_images):
                # reset per-image state
                state["start_time"] = None
                progress_bar.progress(0)
                eta_text.empty()

                status_text.text(f"Generating image {img_i+1}/{num_images} ...")
                try:
                    result = pipe(
                        final_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        callback=diffusers_callback,
                        callback_steps=1,
                        generator=generator
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    status_text.text("Generation failed.")
                    break

                pil_img = result.images[0]
                # optionally watermark (do not modify original pil_img copy for saving raw)
                display_img = pil_img.copy()
                if st.session_state.get("show_wm", True) and watermark_style != "none":
                    display_img = watermark_image(display_img, text="AI-generated (Talrn)", style=watermark_style)
                images.append((pil_img, display_img))

        status_text.text("Generation completed.")

        # display & save results
        with results_container:
            for i, (orig_img, display_img) in enumerate(images):
                st.image(display_img, caption=f"Image {i+1}", use_column_width='always')

                # save both formats and metadata
                meta_params = {
                    "user_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "style_preset": chosen_style,
                    "quality_descriptors": chosen_quality,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed_val,
                    "model": MODEL_ID
                }
                paths, meta_file = save_image_and_meta_formats(
                    orig_img if not st.session_state.get("show_wm", True) else display_img,
                    final_prompt,
                    meta_params,
                    out_dir
                )

                # provide downloads in-memory (PNG + JPEG)
                for fmt, path in paths.items():
                    with open(path, "rb") as f:
                        data = f.read()
                    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
                    st.download_button(
                        label=f"Download Image {i+1} ({fmt})",
                        data=data,
                        file_name=os.path.basename(path),
                        mime=mime
                    )

        st.success(f"Saved {len(images)} images to {out_dir}. Metadata at {meta_file}")
        progress_bar.progress(100)
        eta_text.empty()
