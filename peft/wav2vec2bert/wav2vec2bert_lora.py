from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Wav2Vec2BertModel, Wav2Vec2BertProcessor

MODEL_NAME = "facebook/w2v-bert-2.0"


model = Wav2Vec2BertModel.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    init_lora_weights="gaussian",
    r=32,
    lora_alpha=64,
    target_modules=["linear_q", "linear_v"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_NAME)
