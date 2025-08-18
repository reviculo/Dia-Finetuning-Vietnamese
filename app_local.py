import scipy.signal
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
from pathlib import Path
import gradio as gr
import safetensors.torch as st
from safetensors.torch import load_file as safe_load_file  # <-- THÊM

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
# --- Hết patch ---

# --- Quét folder chứa tất cả checkpoint .pth ---
CKPT_DIR = Path("dia")
ckpt_files = sorted([str(p) for p in CKPT_DIR.glob("*.safetensors")] +
                    [str(p) for p in CKPT_DIR.glob("*.pt")] +
                    [str(p) for p in CKPT_DIR.glob("*.pth")])

if not ckpt_files:
    raise RuntimeError(f"No checkpoints found in {CKPT_DIR}")

# Dropdown để chọn checkpoint
checkpoint_selector = gr.Dropdown(
    choices=ckpt_files,
    value=ckpt_files[0],      # mặc định chọn file đầu tiên
    label="Select Checkpoint"
)

# Textbox để hiển thị trạng thái load model
status = gr.Textbox(label="Model Status", interactive=False)

# Hàm để load model lại mỗi khi chọn checkpoint khác
def reload_model(ckpt_path):
    global model

    # Nếu là .safetensors: đọc an toàn bằng safetensors rồi convert sang .pt tạm
    # để giữ nguyên luồng nạp của Dia.from_local (vốn dùng torch.load bên trong).
    tmp_pt_path = None
    if str(ckpt_path).endswith(".safetensors"):
        state_dict = safe_load_file(str(ckpt_path), device="cpu")  # KHÔNG dùng pickle
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp_pt_path = tmp.name
        tmp.close()
        torch.save(state_dict, tmp_pt_path)  # chuyển dạng sang .pt (pickle) để Dia.from_local nạp

        ckpt_to_load = tmp_pt_path
    else:
        ckpt_to_load = ckpt_path

    # Nạp model theo config + checkpoint đã chuẩn hóa ở trên
    model = Dia.from_local(
        config_path=args.config,
        checkpoint_path=ckpt_to_load,
        device=device
    )

    # Dọn file tạm (nếu có)
    if tmp_pt_path is not None:
        try:
            Path(tmp_pt_path).unlink(missing_ok=True)
        except Exception:
            pass

    # FP16 nếu cần
    if args.half and hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model.model = model.model.half()

    # torch.compile nếu cần
    if args.compile and hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model.model = torch.compile(model.model, backend="inductor")

    # ✅ BẮT BUỘC GÁN DAC SAU KHI LOAD
    dac_model = dac.DAC.load(dac.utils.download())
    dac_model = dac_model.to(device)
    model.dac_model = dac_model

    return f"Loaded checkpoint: {Path(ckpt_path).name}"

# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument(
    "--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')"
)
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument("--local_ckpt", type=str, default="dia/model.safetensors", help="path to your local checkpoint")
parser.add_argument("--config", type=str, default="dia/config_inference.json", help="path to your inference")
parser.add_argument("--half", type=bool, default=False, help="load model in fp16")
parser.add_argument("--compile", type=bool, default=False, help="torch compile model")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")

try:
    cfg = DiaConfig.load("dia/config.json")

    ptmodel = DiaModel(cfg)
    if args.half:
        ptmodel = ptmodel.half()
    if args.compile:
        ptmodel = torch.compile(ptmodel, backend="inductor")

    state = _torch_load_compat(args.local_ckpt, map_location="cpu")
    if "model" in state:
        ptmodel.load_state_dict(state["model"])
    else:
        ptmodel.load_state_dict(state)
    print("✅ Model loaded successfully! Please wait...")
    ptmodel = ptmodel.to(device).eval()

    model = Dia(cfg, device)
    model.model = ptmodel

    # ✅ Cần thiết để voice cloning hoạt động
    dac_model = dac.DAC.load(dac.utils.download())
    dac_model = dac_model.to(device)
    model.dac_model = dac_model

except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

def trim_silence(audio: np.ndarray, threshold: float = 0.01, margin: int = 1000) -> np.ndarray:
    """
    Cắt bỏ vùng im lặng ở đầu và cuối audio numpy.
    - `threshold`: ngưỡng biên độ để coi là 'có tiếng'
    - `margin`: giữ lại một ít trước và sau vùng có tiếng (tính theo mẫu)
    """
    abs_audio = np.abs(audio)
    non_silent_indices = np.where(abs_audio > threshold)[0]

    if non_silent_indices.size == 0:
        return audio  # Nếu hoàn toàn im lặng

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


    print(f"[DEBUG] max_new_tokens = {max_new_tokens}")
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device
    # ✅ Reset conditioning cache nếu có
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
            # Resample nếu không phải 44100
            if sr != 44100:
                try:
                    import librosa
                    # librosa yêu cầu float32 input
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=44100)
                    sr = 44100
                except Exception as e:
                    raise gr.Error(f"Resampling failed: {e}")

            # Check if audio_data is valid
            if (
                audio_data is None
                or audio_data.size == 0
                or np.max(np.abs(audio_data)) < 1e-4  # quá nhỏ
                or len(audio_data) < 1000             # quá ngắn (tương đương ~23ms ở 44.1kHz)
            ):
                gr.Warning("Audio prompt quá ngắn hoặc không hợp lệ sau xử lý. Đã bỏ qua prompt.")
                audio_prompt_input = None
                prompt_path_for_generate = None
                temp_audio_prompt_path = None
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".wav", delete=False
                ) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(
                            f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion."
                        )
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f"Failed to convert audio prompt to float32: {conv_e}"
                            )

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0]
                                if audio_data.shape[0] < audio_data.shape[1]
                                else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(
                            audio_data
                        )  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})"
                        )
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation

        start_time = time.time()

        # Use torch.inference_mode() context manager for the generation call
        # 3. Xử lý văn bản dài bằng cách tách câu
        # --- Nếu CÓ audio prompt: xử lý nguyên khối, không chia câu ---
        if prompt_path_for_generate:
            chunks = [text_input.strip()]
            print("[INFO] Đã phát hiện audio prompt - xử lý toàn bộ văn bản như một đoạn duy nhất.")
        else:
            # --- Nếu KHÔNG có audio prompt: chia theo speaker và câu như bình thường ---
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
        
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        if current_speaker:
                            chunks.append(f"[{current_speaker}] {sent}")
                        else:
                            chunks.append(sent)
            print(f"[INFO] Văn bản được chia thành {len(chunks)} đoạn theo speaker/câu.")   
        
        # Sinh từng đoạn nhỏ và nối lại
        generated_segments = []
        with torch.inference_mode():
            print(f"📄 Văn bản dài, tách thành {len(chunks)} đoạn.")
            for idx, chunk in enumerate(chunks):
                print(f"[Đoạn {idx+1}] {chunk}")
                
                text_for_model = chunk  # channel đã nằm trong chunk rồi, không cần thêm
        
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
                    segment = trim_silence(segment, threshold=0.01, margin=1000)
                    # ✅ Thêm khoảng nghỉ ngắn vào cuối mỗi câu để nghe giống người
                    pause = np.zeros(int(0.5 * 44100), dtype=np.float32)  # 0.25s pause
                    segment = np.concatenate([segment, pause])
                    generated_segments.append(segment)
        
        # Ghép toàn bộ đoạn lại (có thể thêm silence nếu cần)
        if generated_segments:
            combined = []
            group = []
            for i, seg in enumerate(generated_segments):
                group.append(seg)
                if len(group) == 2 or i == len(generated_segments) - 1:
                    # Ghép 2 câu lại thành 1 đoạn
                    if len(group) == 2:
                        merged = np.concatenate(group)
                    else:
                        merged = group[0]
                    combined.append(merged)
                    group = []
            output_audio_np = np.concatenate(combined)

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(
                    f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                )
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(
                f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}"
            )

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        raise gr.Error(f"Inference failed: {e}")

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}"
                )
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}"
                )

    return output_audio


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = ""
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")
    # ← chèn selector/check status
    with gr.Row():
        checkpoint_selector.render()
        status.render()
    checkpoint_selector.change(
        fn=reload_model,
        inputs=[checkpoint_selector],
        outputs=[status]
    )
    init_msg = reload_model(checkpoint_selector.value)
    status.value = init_msg
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
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
                    value=3072,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
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
        #
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



    # Link button click to function
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
        outputs=[audio_output],  # Add status_output here if using it
        api_name="generate_audio",
    )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=True, server_name="0.0.0.0")

