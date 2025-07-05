# 🇻🇳 Dia TTS – Fine-Tuning Tiếng Việt

Dia là mô hình chuyển văn bản thành giọng nói (TTS) 1.6B tham số, được phát triển bởi Nari Labs. Dự án này tinh chỉnh lại Dia để hỗ trợ **tiếng Việt**, tạo ra giọng nói tự nhiên, giàu cảm xúc, hỗ trợ đa nhân vật và có thể clone giọng.

---

## 🚀 Điểm nổi bật

- ✅ Fine-tune mô hình Dia 1.6B với dữ liệu tiếng Việt
- ✅ Hỗ trợ voice cloning đa nhân vật (giọng miền Bắc, Nam, nữ, trẻ em…)
- ✅ Tùy chỉnh chất lượng sinh âm qua `temperature`, `top_p`, `cfg_scale`, v.v.
- ✅ Giao diện Gradio thân thiện, có thể nhập tag như `[01]`, `[Nam-Bac-QuanSu]`, hoặc `[KienThucQuanSu]`
- ✅ Hỗ trợ dataset từ Hugging Face hoặc local `.csv + audio`
- ✅ Tăng tốc bằng `torch.compile`, `bfloat16`, 8-bit optimizer

---

## 🧠 Cấu trúc dự án

| File / Folder         | Mô tả ngắn                                           |
|-----------------------|------------------------------------------------------|
| `app_local.py`        | Giao diện Gradio sinh giọng nói                     |
| `finetune.py`         | Pipeline huấn luyện mô hình với tiếng Việt         |
| `convert_ckpt.py`     | Chuyển checkpoint từ fp16 sang fp32                |
| `config.json`         | Cấu hình kiến trúc mô hình & token đặc biệt         |
| `config_inference.json` | Dành cho sinh giọng inference (float32)          |
| `model.py / layers.py`| Cấu trúc mô hình Dia (Transformer)                 |
| `dataset.py`          | Tiền xử lý dữ liệu từ CSV/HF Dataset                |

---

## 🛠️ Cài đặt

```bash
git clone https://github.com/TuananhCR/dia-ft-vn.git
cd dia-vietnamese
python -m venv .venv
source .venv/bin/activate
pip install -e .
