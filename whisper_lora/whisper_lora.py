from peft import LoraConfig, get_peft_model, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from datasets import load_dataset, DatasetDict

# our env variables
MODEL_NAME = "openai/whisper-tiny"
OUTPUT_DIR = "/exp/..."

# define training hyperparameters and settings (look into this again to fully understand and adapt it to your usecase)
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=...,
    warmup_steps=0,
    num_train_epochs=...,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_first_step=True,
    logging_nan_inf_filter=False,
    eval_steps=500,
    report_to["wandb"],
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=129,
    logging_steps=1,
    remove_unused_columns=False,
    label_names=["labels"],
)

# instantiate processor and model from env variable
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# load dataset
ds = DatasetDict()
ds["train"] = load_dataset("google/fleurs", "is_is", split="train+validation", trust_remote_code=True)
ds["test"] = load_dataset("google/fleurs", "is_is", split="test", trust_remote_code=True)

# setup lora configuration and instantiate model with it
lora_config = LoraConfig(
    init_lora_weights="pissa",
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# setup trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collater=...,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# train
trainer.train()
