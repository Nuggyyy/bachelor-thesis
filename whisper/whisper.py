from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset
from evaluate import load

MODEL_NAME = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")

ds = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    
    batch["reference"] = processor.tokenizer.normalize(batch["transcription"])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer.normalize(transcription)

    return batch

result = ds.map(map_to_pred)

wer = load("wer")
print(f"Word Error Rate: {round(wer.compute(references=result["reference"], predictions=result["prediction"]) * 100, 2)}%")
