import torch
from qwen_asr import Qwen3ASRModel
from peft import PeftModel
import librosa
import numpy as np
from jiwer import wer
import re

# Change these paths as needed
base_model_name = "Qwen/Qwen3-ASR-0.6B"
adapter_path = "./exp/test/checkpoint-376/"


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def load_model_with_adapter(base_name: str, adapter_path: str, device: str = "cuda:0"):
    # Load the Qwen ASR wrapper used during training so processor and model match
    asr = Qwen3ASRModel.from_pretrained(
        base_name,
        device_map=device,
        dtype=torch.float16,
    )

    processor = asr.processor
    base_model = asr.model

    # Attach LoRA adapter; ensure dtype/device alignment
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    return model, processor


model, processor = load_model_with_adapter(base_model_name, adapter_path, device="cuda:0")


def transcribe(audio_path: str, language: str | None = None):
    # Load and resample audio to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = np.array(audio, dtype=np.float32)

    # Build prefix text expected by the Qwen processor
    prompt = f"language {language}" if language else ""
    prefix_msg = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template([prefix_msg], add_generation_prompt=True, tokenize=False)[0]

    # Prepare inputs via processor and move tensors to model device
    inputs = processor(
        text=prefix_text,
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Cast floating tensors to model dtype to avoid mismatch (float32 vs float16)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(dtype=model_dtype)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    # Normalize generation output to a list-of-sequences (list[list[int]] or tensor)
    seq = None
    if isinstance(outputs, tuple):
        seq = outputs[0]
    elif isinstance(outputs, dict):
        seq = outputs.get("sequences", outputs)
    else:
        seq = outputs

    # Convert tensors to CPU-side python lists for tokenizer
    try:
        if torch.is_tensor(seq):
            seq_cpu = seq.detach().cpu().tolist()
        else:
            seq_cpu = [s.detach().cpu().tolist() if torch.is_tensor(s) else s for s in seq]
    except Exception:
        # Fallback: try to coerce directly
        seq_cpu = seq

    try:
        transcription = processor.batch_decode(seq_cpu, skip_special_tokens=True)[0]
    except Exception as e:
        # Provide helpful debug info and re-raise
        raise RuntimeError(
            f"Failed to batch_decode generation output. output_type={type(outputs)}; "
            f"seq_type={type(seq)}; seq_cpu_sample_type={type(seq_cpu[0]) if isinstance(seq_cpu, (list, tuple)) and len(seq_cpu)>0 else type(seq_cpu)}; "
            f"error={e}"
        ) from e

    return transcription


#text = transcribe("Strong_Irish_Accent_Rambling.wav", language="English")
text = transcribe("test_audio.wav", language="English")
text = re.sub("system\nlanguage English\nuser\n\nassistant\n", "", text)

base_model = Qwen3ASRModel.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
results = base_model.transcribe(
    audio="test_audio.wav",
    language="English",
)

# extract transcription from ASR output
base_model_text = results[0].text

references = '''
Now, I don't mind a bit of a breeze. If anything, I prefer it, but thon was aggressive. And I says to myself, says I, Colm, this is no day for a do.
For when the bride arrived, and as I say, by this stage the wind was fierce. I've never heard wind like it, howling like a banshee it was.
So the poor girl, the bride now this is, she arrives anyway, and isn't she no sooner out of the car than she's lifted up in the air, like a paperdoll, and blown into a flowerbed.
'''
references = "Hello, this is a test. I am curious to see how good the word error rate is now with an American accent."
peft_references = re.sub(r"[,\.]", "", references)
print(f"peft:\n{text}")
print(f"base:\n{base_model_text}")
print(f"reference:\n{peft_references.lower()}")
print(f"WER peft: {round(wer(peft_references.lower(), text)*100)}%")
print(f"WER base: {round(wer(references, base_model_text)*100)}%")
