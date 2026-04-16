from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
import torch
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np

# our env variables
MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
OUTPUT_DIR = "./exp/test"
DATASET_NAME = "google/fleurs"

assert torch.cuda.is_available(), "No GPU found!"

# instantiate processor and model from env variable
device = torch.device("cuda")
processor = Qwen3ASRProcessor.from_pretrained(MODEL_NAME)
model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="cuda:0", dtype=torch.float16,)

# load and process dataset
ds = DatasetDict()
ds["train"] = load_dataset(DATASET_NAME, "is_is", split="train+validation", trust_remote_code=True)
ds["test"] = load_dataset(DATASET_NAME, "is_is", split="test", trust_remote_code=True)
def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["transcription"],
    )
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    example["input_ids"] = example["input_ids"][0]
    example["input_features"] = example["input_features"][0]

    # sanity check
    vocab_size = len(processor.tokenizer)
    ids = example["input_ids"]
    bad = [id for id in ids if id < 0 or id >= vocab_size]
    if bad:
        print(f"BAD TOKEN IDs found: {bad}, vocab_size={vocab_size}")
        print(f"Transcription: {example.get('transcription', '?')}")

    return example

ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)
def is_audio_in_length_range(length):
    return length < 30.0
ds["train"] = ds["train"].filter(is_audio_in_length_range, input_columns=["input_length"])

# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": np.array(feature["input_features"]).T} for feature in features
        ]

        batch = self.processor.feature_extractor.pad(input_features, padding=True, return_tensors="pt")

        label_features = [{"input_ids": feature["input_ids"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.audio_bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["input_ids"] = labels

        if "attention_mask" in batch and "feature_attention_mask" not in batch:
            batch["feature_attention_mask"] = batch.pop("attention_mask")
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# evaluation metric
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [processor.tokenizer.normalize(pred) for pred in pred_str]
    label_str_norm = [processor.tokenizer.normalize(label) for label in label_str]

    pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

# received _forward_unimplemented() error and found this on a lora finetuning on hf
def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Không thể patch hàm forward vì thiếu module `.thinker.forward`. "
            "Model của bạn chưa chuẩn."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    # Patch for PEFT LoRA compatibility
    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()
    cls.get_input_embeddings = get_input_embeddings

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)
    cls.set_input_embeddings = set_input_embeddings

    cls._forward_patched = True
patch_outer_forward(model)

# 1. Resize embeddings to match the tokenizer
#model.resize_token_embeddings(len(processor.tokenizer))
# 2. Update the thinker's configuration to be sure (since you are patching)
#model.thinker.config.vocab_size = len(processor.tokenizer)

# setup lora configuration and instantiate model with it
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    logging_strategy="steps",
    logging_first_step=True,
    logging_nan_inf_filter=False,
    logging_steps=10,
    report_to=["tensorboard"],
    fp16=True,
    fp16_full_eval=True,
    remove_unused_columns=False,
    label_names=["input_ids"],
)

# setup trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

model.base_model.model.thinker.resize_token_embeddings(len(processor.tokenizer))

# train
trainer.train()
