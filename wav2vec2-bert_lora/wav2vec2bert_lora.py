from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, Wav2Vec2BertProcessor

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

try:
    # `facebook/w2v-bert-2.0` is a pretraining checkpoint and typically does not include
    # a tokenizer/vocab. In that case, a full `Wav2Vec2BertProcessor` cannot be loaded.
    processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_NAME)
except TypeError as e:
    # The failure mode we want to recover from is: tokenizer init tries to `open(vocab_file)`
    # where `vocab_file` is None, raising:
    #   TypeError: expected str, bytes or os.PathLike object, not NoneType
    msg = str(e)
    if "os.PathLike" not in msg or "NoneType" not in msg:
        raise
    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
