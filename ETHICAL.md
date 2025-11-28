\# Ethical AI Usage Guidelines



This project uses an open-source text-to-image generative model (Stable Diffusion). 

To ensure responsible and ethical AI usage, the following guidelines must be followed:



---


\## 1. Prohibited Content



The model must \*\*not\*\* be used to generate any of the following:



\- Sexually explicit or adult content  

\- Child exploitation or minors in inappropriate situations  

\- Graphic violence or gore  

\- Hate speech, harassment, or harmful stereotypes  

\- Deepfakes of real people without consent  

\- Illegal activities or harmful instructions  



The application includes a basic keyword blocklist, but users are responsible for ethical usage.



---



\## 2. Watermarking \& Transparency



All generated images \*\*should be watermarked\*\* to indicate AI origin.  

This is enabled by default in the application:

This ensures transparency when sharing images publicly.

---

## 3. Content Filtering

The safety checker in Stable Diffusion is disabled **only for local testing purposes**.  
For public-facing or production systems:

✔ Re-enable diffusion safety checker  
✔ Add additional text filtering  
✔ Consider using CLIP or moderation ML models to classify unsafe prompts  

---

## 4. Copyright & Fair Use

- Do not recreate copyrighted characters, logos, or trademarked artwork.  
- Avoid prompts that replicate a living artist’s style exactly (e.g., “in the style of XYZ”).  
- Use the model responsibly, respecting intellectual property rights.

---

## 5. Bias and Hallucinations

AI models may produce:

- Biased representations  
- Incorrect or misleading details  
- Object deformities or unrealistic structures  

Users must review outputs carefully and avoid presenting AI-generated images as factual.

---

## 6. User Responsibility

By using this project, the user agrees to:

- Follow ethical guideline
- Not use the system for harmful purposes  
- Clearly disclose AI-generated content when published online  

---

## 7. Developer Disclaimer

This project is developed **for educational and research purposes only** as part of the Talrn ML Internship assessment.  
The developer is **not responsible** for misuse of the model.

---





