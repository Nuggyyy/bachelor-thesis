from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset
from jiwer import wer

MODEL_NAME = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

ds = load_dataset("google/fleurs", "is_is", split="test", trust_remote_code=True)

error = float()
count = 0
for item in ds:
    count += 1
    
    input_features = processor(item["audio"]["array"], sampling_rate=item["audio"]["sampling_rate"], return_tensors="pt").input_features
    
    reference = item["transcription"]

    predicted_ids = model.generate(input_features)
    hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    error += wer(reference, hypothesis)

final_wer = error / count
print(f"Word Error Rate: {round(final_wer * 100, 2)}%")
