from peft import LoraConfig

lora_config = LoraConfig(
    init_lora_weights="gaussian"
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
