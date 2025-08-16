# 🇻🇳 Dia TTS – Fine-Tuning Vietnamese

High‑quality Vietnamese speech generation 44.1 kHz on top of Nari Labs’ DIA 1.6B. This repo provides an unofficial fine‑tune enabling natural Vietnamese with controllable style, multi‑speaker accents, and a friendly Gradio demo.

---

⚠️ Status: Community release. Upstream DIA currently ships English generation; this project adds Vietnamese via fine‑tuning. Follow ethical use guidelines below.
- Maintainer: Tuan Anh — AI/ML Researcher @ Appota SRD (R&D Department)
- Compute: Trained and developed on Appota’s server infrastructure



https://github.com/user-attachments/assets/8e5604eb-e3b7-4cee-99e6-f18dfd546788



## Features

- ✅ Fine-tune model Dia 1.6B with Vietnamese Dataset
- ✅ Support single speaker and multispeaker with various Vietnamese accent ( Nort-male, South-male, North-female and South-female)
- ✅ Adjusting voice generate by `temperature`, `top_p`, `cfg_scale`, etc.
- ✅ Friendly Gradio Inference
- ✅ Speed up by `torch.compile`, `bfloat16`, 8-bit optimizer
---
- You can try demo at : https://huggingface.co/spaces/cosrigel/Dia-Vietnamese
- You can use our finetune model at : https://huggingface.co/cosrigel/dia-finetuning-vnese
---

## Data Preparation for Finetuning model
- Audio: mono 44.1 kHz WAV/FLAC; per‑utterance 3–20 s; peak‑normalized.
- Dataset finetuned : capleaf/viVoice
- Total duration: 1,016.97 hours

### Training Configuration:
- Base model : nari-labs/Dia-1.6B
- GPU : NVIDIA RTX A6000
- You can use our checkpoint to use the inference at : https://huggingface.co/cosrigel/dia-finetuning-vnese

## Inference Tips (Vietnamese)
- Transcripts: begin with [01] or [KienThucQuanSu] then text
- For example: [KienThucQuanSu] Thủ tướng cũng yêu cầu các Bộ, cơ quan trung ương, địa phương tăng cường công tác thanh tra, kiểm tra việc sắp xếp, xử lý tài sản trước, trong và sau khi sắp xếp tổ chức bộ máy, sắp xếp đơn vị hành chính.
- You can lookup the speaker ID in speaker table ID which is already existed in Gradio Inference
<img width="1545" height="903" alt="Screenshot 2025-08-16 at 09 53 21" src="https://github.com/user-attachments/assets/42a24781-0aaf-402d-aa37-901f0046c9cc" />

## Future Feature Improve
- ☐ Increase the quality of voice-cloning
- ☐ Add emotion to dataset and model so you can you emotion's tag like : [cười], [khóc], [ho],...
- ☐ Adjust the voice of multispeaker so they can sound like they're all in one room speaking to each other.

## Ethics & Responsible Use
- Obtain consent for any real person’s voice.
- Disclose synthetic audio in production settings.
- No impersonation, harassment, or deceptive content.

## 🛠️ Installation
```bash
git clone https://github.com/TuananhCR/Dia-Finetuning-Vietnamese
cd dia-vietnamese
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage Example
```bash
python finetune.py \
  --config configs/config.json \
  --dataset  \
  --hub_model nari-labs/Dia-1.6B \
  --run_name dia_vietnamese_experiment \
  --output_dir ./checkpoints
```

## Acknowledgements
- Appota SRD (R&D Department) — compute & infrastructure support for training and development
- Nari Labs – DIA (architecture & checkpoints)
- Descript Audio Codec (DAC) for discrete audio tokens
- Hugging Face Transformers/PEFT/Accelerate

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
