# Semantic & Philosophical Text Analyzer
Most tools label emotion. Chatbots invent depth.
This one reads for **tone, subtext, and philosophical pressure** — and tries not to overclaim.
> *Today my mother died. Or maybe yesterday, I don’t know…*
A shallow system hears *death* → *grief*.  
A careful one hears **numbness and estrangement**.
That difference is the project.
---
**What it returns**
- A short philosophical insight  
- An emotion spectrum (felt stance, not just plot keywords)  
- Feedback (Helpful / Unhelpful) that quietly improves later readings  
**Stack:** Streamlit · Gemini · GoEmotions · literary reference memory · Supabase feedback  
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
It returns a short philosophical insight and an emotion spectrum shaped by what the voice is doing, not only by what happens in the plot. When a reading helps, you can say so — and later readings grow from that.
