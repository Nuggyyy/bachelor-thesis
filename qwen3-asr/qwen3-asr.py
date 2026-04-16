import torch
from qwen_asr import Qwen3ASRModel
from datasets import load_dataset
from evaluate import load
from jiwer import wer

MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
DATASET_NAME = "google/fleurs"

model = Qwen3ASRModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

ds = load_dataset(
    DATASET_NAME,
    "en_us",
    split="test",
    trust_remote_code=True,
)

audios = [(audio["audio"]["array"], audio["audio"]["sampling_rate"]) for audio in ds]

results = model.transcribe(
    audio=audios
)

predictions = [result.text for result in results]
references = [audio["transcription"] for audio in ds]

print(f"Word Error Rate: {round(wer(references, predictions) * 100, 2)}%")
