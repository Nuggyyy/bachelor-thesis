from peft import LoraConfig, get_peft_model, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer

MODEL_NAME = "openai/whisper-tiny"


model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    init_lora_weights="gaussian",
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# this is used for processing the dataset
processor = WhisperProcessor.from_pretrained(MODEL_NAME,
                                             task="transcribe")
