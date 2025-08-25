import re
import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from dia.model import Dia
from dia.config import DiaConfig
from dia.layers import DiaModel
import dac
import safetensors.torch as st
from safetensors.torch import load_file as safe_load_file

# === NEW: API imports ===
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import base64
import io
import uvicorn

# --- Patch PyTorch 2.6: đảm bảo torch.load không dùng weights_only=True mặc định ---
_orig_torch_load = torch.load
def _torch_load_compat(path, *args, **kwargs):
    """
    Load checkpoint tương thích cả .pt/.pth và .safetensors
    """
    if isinstance(path, str) and path.endswith(".safetensors"):
        return st.load_file(path)
    else:
        return _orig_torch_load(path, *args, **kwargs)
torch.load = _torch_load_compat

# --- Quét folder chứa tất cả checkpoint .pth ---
CKPT_DIR = Path("dia")
ckpt_files = sorted([str(p) for p in CKPT_DIR.glob("*.safetensors")] +
                    [str(p) for p in CKPT_DIR.glob("*.pt")] +
                    [str(p) for p in CKPT_DIR.glob("*.pth")])

if not ckpt_files:
    raise RuntimeError(f"No checkpoints found in {CKPT_DIR}")

# Textbox để hiển thị trạng thái load model
status = gr.Textbox(label="Model Status", interactive=False)

# Đặt global ở gần đầu file (nếu chưa có)
model = None
dac_model = None

def load_model_once():
    """
    Load model duy nhất một lần khi khởi động.
    Giữ logic .safetensors -> .pt, half/compile chỉ trên CUDA, gắn DAC một lần.
    """
    global model, dac_model

    if model is not None:
        return f"Model already loaded on {device}"

    ckpt_path = Path(args.local_ckpt)
    tmp_pt_path = None

    # Nếu checkpoint là .safetensors, chuyển tạm sang .pt để tương thích torch.load
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError("Chưa cài safetensors, không thể nạp .safetensors") from e

        state_dict = safe_load_file(str(ckpt_path), device="cpu")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_pt_path = tmp.name
        tmp.close()
        torch.save(state_dict, tmp_pt_path)
        ckpt_to_load = tmp_pt_path
    else:
        ckpt_to_load = str(ckpt_path)

    # Gọi Dia.from_local với compute_dtype nếu bạn đã khai báo ở Bước 2
    kwargs = dict(
        config_path=args.config,
        checkpoint_path=ckpt_to_load,
        device=device,
    )
    if "compute_dtype" in globals():
        kwargs["compute_dtype"] = compute_dtype

    model_local = Dia.from_local(**kwargs)

    # Xoá file tạm (nếu có)
    if tmp_pt_path is not None:
        try:
            Path(tmp_pt_path).unlink(missing_ok=True)
        except Exception:
            pass

    # half / compile CHỈ trên CUDA
    if getattr(args, "half", False) and device.type == "cuda" and hasattr(model_local, "model"):
        model_local.model = model_local.model.half()

    if getattr(args, "compile", False) and device.type == "cuda" and hasattr(model_local, "model"):
        model_local.model = torch.compile(model_local.model, backend="inductor")

    # Gắn DAC đúng device — chỉ load một lần cho toàn app
    if dac_model is None:
        _dac = dac.DAC.load(dac.utils.download()).to(device)
        dac_model_local = _dac
        globals()["dac_model"] = dac_model_local
    else:
        dac_model_local = dac_model

    model_local.dac_model = dac_model_local

    # Xuất ra global
    model = model_local
    return f"Loaded checkpoint: {ckpt_path.name} on {device}"

# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio + API for Nari/Dia TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="(unused in API mode)")
parser.add_argument("--local_ckpt", type=str, default="dia/model.safetensors", help="path to your local checkpoint")
parser.add_argument("--config", type=str, default="dia/config_inference.json", help="path to your inference")
parser.add_argument("--half", type=bool, default=False, help="load model in fp16")
parser.add_argument("--compile", type=bool, default=False, help="torch compile model")
# === NEW: API host/port ===
parser.add_argument("--api_host", type=str, default="0.0.0.0", help="API host")
parser.add_argument("--api_port", type=int, default=7860, help="API port")
parser.add_argument("--mount_path", type=str, default="/ui", help="Gradio UI mount path")

args = parser.parse_args()

# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        from torch.nn.attention import sdpa_kernel
        sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass

# Dtype cho Dia (app.py dùng chuỗi cho compute_dtype)
_dtype_map = {"cpu": "float32", "mps": "float32", "cuda": "float16"}
compute_dtype = _dtype_map.get(device.type, "float16")
print(f"compute_dtype for Dia: {compute_dtype}")

# Load Nari model and config (kept from your original)
print("Loading Nari model...")

try:
    cfg = DiaConfig.load(args.config if getattr(args, "config", None) else "dia/config.json")

    ptmodel = DiaModel(cfg)

    # ✅ half chỉ trên CUDA
    if getattr(args, "half", False) and device.type == "cuda":
        ptmodel = ptmodel.half()

    # ✅ compile chỉ trên CUDA
    if getattr(args, "compile", False) and device.type == "cuda":
        ptmodel = torch.compile(ptmodel, backend="inductor")

    # Tải state ở CPU rồi chuyển device
    state = _torch_load_compat(args.local_ckpt, map_location="cpu")
    ptmodel.load_state_dict(state["model"] if "model" in state else state, strict=True)

    print("✅ Model loaded successfully! Please wait...")
    ptmodel = ptmodel.to(device).eval()

    model = Dia(cfg, device)
    model.model = ptmodel

    # ✅ DAC đúng device
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model.dac_model = dac_model

except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

def trim_silence(audio: np.ndarray, threshold: float = 0.01, margin: int = 1000) -> np.ndarray:
    abs_audio = np.abs(audio)
    non_silent_indices = np.where(abs_audio > threshold)[0]
    if non_silent_indices.size == 0:
        return audio
    start = max(non_silent_indices[0] - margin, 0)
    end = min(non_silent_indices[-1] + margin, len(audio))
    return audio[start:end]

def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    # (unchanged core from your file)  :contentReference[oaicite:1]{index=1}
    print(f"[DEBUG] max_new_tokens = {max_new_tokens}")
    global model, device
    if hasattr(model, "reset_conditioning"):
        model.reset_conditioning()
        print("[DEBUG] Đã reset conditioning latent voice.")
    elif hasattr(model, "voice_encoder_cache"):
        model.voice_encoder_cache = {}
        print("[DEBUG] Đã xoá voice encoder cache.")
    else:
        print("[DEBUG] Không tìm thấy cơ chế reset conditioning, bỏ qua.")

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None

        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            if sr != 44100:
                try:
                    import librosa
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=44100)
                    sr = 44100
                except Exception as e:
                    raise gr.Error(f"Resampling failed: {e}")

            if (
                audio_data is None
                or audio_data.size == 0
                or np.max(np.abs(audio_data)) < 1e-4
                or len(audio_data) < 1000
            ):
                gr.Warning("Audio prompt quá ngắn hoặc không hợp lệ sau xử lý. Đã bỏ qua prompt.")
                audio_prompt_input = None
                prompt_path_for_generate = None
                temp_audio_prompt_path = None
            else:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis.")
                            audio_data = (
                                audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(audio_data)

                    try:
                        sf.write(temp_audio_prompt_path, audio_data, sr, subtype="FLOAT")
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        start_time = time.time()

        # if prompt_path_for_generate:
            # chunks = [text_input.strip()]
            # print("[INFO] Đã phát hiện audio prompt - xử lý toàn bộ văn bản như một đoạn duy nhất.")
        # else:
            # speaker_blocks = re.split(r'(?=\[[^\]]+\])', text_input.strip())
            # chunks = []
            # current_speaker = None

            # for block in speaker_blocks:
                # block = block.strip()
                # if not block:
                    # continue

                # speaker_match = re.match(r"\[([^\]]+)\]\s*(.*)", block, re.DOTALL)
                # if speaker_match:
                    # current_speaker = speaker_match.group(1)
                    # content = speaker_match.group(2).strip()
                # else:
                    # content = block

                # sentences = re.split(r'(?<=[.!?])\s+', content)
                # for sent in sentences:
                    # sent = sent.strip()
                    # if sent:
                        # if current_speaker:
                            # chunks.append(f"[{current_speaker}] {sent}")
                        # else:
                            # chunks.append(sent)
            # print(f"[INFO] Văn bản được chia thành {len(chunks)} đoạn theo speaker/câu.")

        # ALWAYS split text into speaker-aware sentences, even with audio prompt
        speaker_blocks = re.split(r'(?=\[[^\]]+\])', text_input.strip())

        chunks = []
        current_speaker = None
        for block in speaker_blocks:
            block = block.strip()
            if not block:
                continue

            speaker_match = re.match(r"\[([^\]]+)\]\s*(.*)", block, re.DOTALL)
            if speaker_match:
                current_speaker = speaker_match.group(1)
                content = speaker_match.group(2).strip()
            else:
                content = block

            # Ensure we end with punctuation so the model sees a clear stop
            if content and content[-1] not in ".!?…":
                content += "."

            # Primary sentence split (+ Vietnamese ellipsis)
            sentences = re.split(r'(?<=[\.\!\?…])\s+', content)

            # Secondary split for very long sentences
            _LONG = 180  # chars
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(sent) > _LONG:
                    # Cut on commas/semicolons first, then hard-wrap if still too long
                    subparts = re.split(r'(?<=[,;:])\s+', sent)
                    buf = ""
                    for sp in subparts:
                        if len(buf) + len(sp) + 1 <= _LONG:
                            buf = (buf + " " + sp).strip()
                        else:
                            if current_speaker:
                                chunks.append(f"[{current_speaker}] {buf}")
                            else:
                                chunks.append(buf)
                            buf = sp
                    if buf:
                        if current_speaker:
                            chunks.append(f"[{current_speaker}] {buf}")
                        else:
                            chunks.append(buf)
                else:
                    if current_speaker:
                        chunks.append(f"[{current_speaker}] {sent}")
                    else:
                        chunks.append(sent)

        print(f"[INFO] Văn bản được chia thành {len(chunks)} đoạn theo speaker/câu.")




        generated_segments = []
        with torch.inference_mode():
            print(f"📄 Văn bản dài, tách thành {len(chunks)} đoạn.")
            for idx, chunk in enumerate(chunks):
                print(f"[Đoạn {idx+1}] {chunk}")
                text_for_model = chunk
                segment = model.generate(
                    text_for_model,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    use_cfg_filter=True,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,
                    audio_prompt_path=prompt_path_for_generate,
                )
                if segment is not None and isinstance(segment, np.ndarray):
                    #segment = trim_silence(segment, threshold=0.01, margin=1000)
                    # slightly lower threshold; give the last chunk a bigger margin
                    is_last = (idx == len(chunks) - 1)
                    segment = trim_silence(segment, threshold=0.008, margin=(5000 if is_last else 2000))
                    pause = np.zeros(int(0.5 * 44100), dtype=np.float32)
                    segment = np.concatenate([segment, pause])
                    generated_segments.append(segment)

        if generated_segments:
            combined = []
            group = []
            for i, seg in enumerate(generated_segments):
                group.append(seg)
                if len(group) == 2 or i == len(generated_segments) - 1:
                    merged = np.concatenate(group) if len(group) == 2 else group[0]
                    combined.append(merged)
                    group = []
            output_audio_np = np.concatenate(combined)

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        if output_audio_np is not None:
            output_sr = 44100
            original_len = len(output_audio_np)
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)
            if target_len != original_len and target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (output_sr, resampled_audio_np.astype(np.float32))
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (output_sr, output_audio_np)
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            print(f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}")
        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {e}")

    finally:
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}")
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}")

    return output_audio

# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
default_text = "[HocEnglishOnline] Quick tip: “small wins” nghĩa là thắng lợi nhỏ; ghi lại để não nhận phần thưởng, động lực sẽ tăng."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row():
        status.render()

    init_msg = load_model_once()
    status.value = init_msg

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",
            )
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=3072,
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=35,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )

            gr.Markdown("📌 **Copy tag người nói như `[KienThucQuanSu]` để dán vào văn bản sinh giọng phù hợp.**")

            gr.Markdown("### 🟢 Good Voice Speakers (Rõ, chuẩn, chất lượng cao)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female", "Center Female"],
                value=[
                    ["[KienThucQuanSu]", "[kenhCoVan]", "[HocEnglishOnline]", "[CoBaBinhDuong]", "[PTTH-TRT]"],
                    ["[AnimeRewind.Official]", "[ThePresentWriter]", "[HuynhDuyKhuongofficial]", "[SUCKHOETAMSINH]", ""],
                    ["[BroNub]", "[5PhutCrypto]", "[HuynhLapOfficial]", "[TIN3PHUT]", ""],
                    ["[VuiVe]", "[SachBiQuyetThanhCong]", "[NgamRadio]", "", ""],
                    ["[W2WAnime]", "[BIBITV8888]", "", "", ""],
                    ["[DongMauViet]", "", "", "", ""],
                ],
                interactive=False
            )

            gr.Markdown("### 🟡 Normal Voice Speakers (Dùng được, giọng khá ổn)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female"],
                value=[
                    ["[NhaNhac555]", "[sunhuynpodcast.]", "[MensBay]", "[BoringPPL]"],
                    ["[JVevermind]", "[HocvienBovaGau]", "[Web5Ngay]", "[TULEMIENTAY]"],
                    ["[CosmicWriter]", "[SukiesKitchen]", "[AnhBanThan]", "[HappyHidari]"],
                    ["[RuaNgao]", "[Nhantaidaiviet]", "[PhanTichGame]", "[SpiderumBooks]"],
                    ["[TuanTienTi2911]", "[W2WCartoon]", "", "[HoabinhTVgo]"],
                    ["[CuThongThai]", "[BaodientuVOV]", "", "[RiwayLegal]"],
                    ["[meGAME_Official]", "", "", ""],
                ],
                interactive=False
            )

            gr.Markdown("### 🔴 Weak Voice Speakers (Không nên ưu tiên dùng làm mẫu giọng)")
            gr.Dataframe(
                headers=["North Male", "North Female", "South Male", "South Female"],
                value=[
                    ["[TintucBitcoin247]", "[Xanh24h]", "[MangoVid]", "[TheGioiLaptop]"],
                    ["[ThanhPahm]", "", "[ThaiNhiTV]", "[BachHoaXANHcom]"],
                    ["[VuTruNguyenThuy]", "", "[MeovatcuocsongLNV]", ""],
                    ["[NTNVlogsNguyenThanhNam]", "", "", ""],
                    ["[HIEUROTRONG5PHUT-NTKT]", "", "", ""],
                ],
                interactive=False
            )

    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],
        api_name="generate_audio",
    )

# === NEW: FastAPI — request model & endpoints ===
class SynthesizeRequest(BaseModel):
    text: str
    max_new_tokens: int = 3072
    cfg_scale: float = 3.0
    temperature: float = 1.3
    top_p: float = 0.95
    cfg_filter_top_k: int = 35
    speed_factor: float = 0.94
    # Optional: base64-encoded WAV for reference voice
    audio_prompt_b64: Optional[str] = None  # raw WAV/PCM bytes, base64

app = FastAPI(title="Nari/Dia Vietnamese TTS API", version="1.0.0")

@app.on_event("startup")
def _warmup():
    # ensure model is ready for API calls
    try:
        load_model_once()
    except Exception as e:
        print(f"[startup] model load failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.get("/voices")
def voices():
    # keep this short – you can expand later
    return {"hint": "Use speaker tags like [KienThucQuanSu] in your text."}

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """
    Accepts JSON and returns WAV bytes.
    If audio_prompt_b64 is provided (WAV), it will be used as a reference voice.
    """
    # Decode optional prompt
    audio_tuple: Optional[Tuple[int, np.ndarray]] = None
    if req.audio_prompt_b64:
        try:
            raw = base64.b64decode(req.audio_prompt_b64)
            with io.BytesIO(raw) as bio:
                data, sr = sf.read(bio, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1).astype(np.float32)
            audio_tuple = (int(sr), np.asarray(data, dtype=np.float32))
        except Exception as e:
            raise HTTPException(400, f"Invalid audio_prompt_b64: {e}")

    try:
        sr_out, wav_np = run_inference(
            text_input=req.text,
            audio_prompt_input=audio_tuple,
            max_new_tokens=req.max_new_tokens,
            cfg_scale=req.cfg_scale,
            temperature=req.temperature,
            top_p=req.top_p,
            cfg_filter_top_k=req.cfg_filter_top_k,
            speed_factor=req.speed_factor,
        )
    except gr.Error as ge:
        # convert Gradio-friendly errors into HTTP
        raise HTTPException(400, str(ge))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # Encode WAV to bytes
    buf = io.BytesIO()
    try:
        sf.write(buf, wav_np, sr_out, format="WAV", subtype="PCM_16")
    except Exception as e:
        raise HTTPException(500, f"Failed to encode WAV: {e}")

    headers = {"X-Sample-Rate": str(sr_out)}
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)

# Mount Gradio UI at /ui (configurable)
gr.mount_gradio_app(app, demo, path=args.mount_path)

# --- Launch the unified API server (FastAPI + Gradio at /ui) ---
if __name__ == "__main__":
    print("Starting FastAPI server with Gradio UI mounted...")
    uvicorn.run(app, host=args.api_host, port=args.api_port, workers=1)
