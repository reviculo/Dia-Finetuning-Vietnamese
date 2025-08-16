# 🇻🇳 Dia TTS – Fine-Tuning Vietnamese

Dia is a 1.6B parameter text to speech model created by Nari Labs with 1.6B parameters. Dia directly generates highly realistic dialogue from a transcript. This project has been finetuned for using Vietnamese, creat natural voice and tone control.
---

## Features

- ✅ Fine-tune model Dia 1.6B with Vietnamese Dataset
- ✅ Support single speaker and multispeaker with various Vietnamese accent ( Nort-male, South-male, North-female and South-female)
- ✅ Adjusting voice generate by `temperature`, `top_p`, `cfg_scale`, etc.
- ✅ Friendly Gradio Inference
- ✅ Speed up by `torch.compile`, `bfloat16`, 8-bit optimizer

You can try demo at : https://huggingface.co/spaces/cosrigel/Dia-Vietnamese
---

## Data Preparation for Finetuning model
- Audio: mono 44.1 kHz WAV/FLAC; per‑utterance 3–20 s; peak‑normalized.
- Dataset finetuned : capleaf/viVoice
- Total duration: 1,016.97 hours
# Training Configuration:
- Base model : nari-labs/Dia-1.6B
- GPU : NVIDIA RTX A6000


## Inference Tips (Vietnamese)
Transcripts: begin with [01] or [KienThucQuanSu] then text
For example: [KienThucQuanSu] Thủ tướng cũng yêu cầu các Bộ, cơ quan trung ương, địa phương tăng cường công tác thanh tra, kiểm tra việc sắp xếp, xử lý tài sản trước, trong và sau khi sắp xếp tổ chức bộ máy, sắp xếp đơn vị hành chính.
You can lookup the speaker ID in speaker table ID which is already existed in Gradio Inference
<img width="1545" height="903" alt="Screenshot 2025-08-16 at 09 53 21" src="https://github.com/user-attachments/assets/42a24781-0aaf-402d-aa37-901f0046c9cc" />


## 🛠️ Setup

```bash
git clone https://github.com/TuananhCR/Dia-Finetuning-Vietnamese
cd dia-vietnamese
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
## Citation
```
If you use this work, please cite the upstream DIA model and this repository.
@misc{Dia-Finetuning-Vietnamese,
  title        = {DIA Vietnamese Fine-Tuning} ,
  author       = {Cos Rigel},
  year         = {2025},
  howpublished = {GitHub repository},{Huggingface repository}
  url          = {https://github.com/TuananhCR/Dia-Finetuning-Vietnamese}
}
```
