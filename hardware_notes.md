\# Hardware Notes — CPU \& GPU Installation Guide



This project supports both \*\*CPU-only\*\* systems and \*\*GPU-accelerated\*\* setups.

Below are step-by-step instructions for each environment.



---



\# 1. CPU Installation (Recommended for Testing)



Most users without a GPU can run Stable Diffusion using CPU mode.  

This is slower but requires no special hardware.



\### Install CPU PyTorch

```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu



