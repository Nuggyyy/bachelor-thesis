from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset
from evaluate import load
from jiwer import wer
# our env variables
MODEL_NAME = "openai/whisper-tiny"

# instantiate processor and model from env variable
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")

# load dataset
ds = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)

# function that does all the input processing, model prediction and output processing
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    batch["reference"] = processor.tokenizer.normalize(batch["transcription"])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer.normalize(transcription)

    return batch

# previously defined function is called with the .map method (maybe look into what exactly that does)
result = ds.map(map_to_pred)

# word error rate loading and then calculation which is immediately printed to terminal. could adapt it into a dataset for later
print(f"Word Error Rate: {round(wer(result["reference"], result["prediction"]) * 100, 2)}%")
