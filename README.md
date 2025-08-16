# 🇻🇳 Dia TTS – Fine-Tuning Vietnamese

Dia is a 1.6B parameter text to speech model created by Nari Labs with 1.6B parameters. Dia directly generates highly realistic dialogue from a transcript. This project has been finetuned for using Vietnamese, creat natural voice and tone control.
---

## 🚀 Features

- ✅ Fine-tune model Dia 1.6B with Vietnamese Dataset
- ✅ Support single speaker and multispeaker with various Vietnamese accent ( Nort-male, South-male, North-female and South-female )
- ✅ Adjusting voice generate by `temperature`, `top_p`, `cfg_scale`, etc.
- ✅ Friendly Gradio Inference
- ✅ Speed up by `torch.compile`, `bfloat16`, 8-bit optimizer

---

## 🛠️ Setup

```bash
git clone https://github.com/TuananhCR/Dia-Finetuning-Vietnamese
cd dia-vietnamese
python -m venv .venv
source .venv/bin/activate
pip install -e .
