from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer

MODEL_NAME = "openai/whisper-tiny"


model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

