from peft import RandLoraConfig, get_peft_model
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Trainer,
    TrainingArguments,
)
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from jiwer import wer
from random import randint

#SCRIPT_DIR = Path(__file__).parent
MODEL_NAME = "facebook/w2v-bert-2.0"
OUTPUT_DIR = "./exp/javanese"
DATASET_NAME = "google/fleurs"

assert torch.cuda.is_available(), "No GPU found!"

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = Wav2Vec2CTCTokenizer("javanese.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

ds = DatasetDict()
ds["train"] = load_dataset(DATASET_NAME, "jv_id", split="train+validation")
ds["test"]  = load_dataset(DATASET_NAME, "jv_id", split="test")

def prepare_dataset(example):
    audio = example["audio"]
    result = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["transcription"],
    )
    result["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return result

ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)
ds["train"] = ds["train"].filter(lambda x: x < 30.0, input_columns=["input_length"])

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

def compute_metrics(pred):
    pred_ids = pred.predictions
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
 
    index = randint(0, len(pred_str) - 3)
    # Print a few examples for debugging (helps explain huge WERs)
    for ref, hyp in list(zip(label_str, pred_str))[index:index + 3]:
        print("--- Eval sample ---")
        print("REF:", repr(ref))
        print("HYP:", repr(hyp))

    return {"wer": 100 * wer(label_str, pred_str)}

model = Wav2Vec2BertForCTC.from_pretrained(
    MODEL_NAME,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
model.config.use_cache = False

randlora_config = RandLoraConfig(
    r=32,
    target_modules=["linear_q", "linear_v", "linear_k", "linear_out"],
    randlora_dropout=0.1,
    modules_to_save=["lm_head"]
)
model = get_peft_model(model, lora_config)
model.get_input_embeddings = lambda: model.base_model.model.wav2vec2_bert.feature_projection.projection
model.get_output_embeddings = lambda: model.base_model.model.lm_head
model.base_model.save_embedding_layers = True
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # smaller per-device batch + accumulation to keep effective batch stable on limited data / GPU
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch size = 8 * 4 = 32
    learning_rate=1e-3,
    warmup_steps=200,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    num_train_epochs=12,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_on_start=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=4,
    logging_first_step=True,
    logging_steps=10,
    report_to=["tensorboard"],
    fp16=True,
    fp16_full_eval=True,
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adamw_torch_fused",
    weight_decay=0.01,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

def preprocess_logits_for_metrics(logits, labels):
    # logits can be a tensor or a tuple (model outputs). Take first element if tuple.
    if isinstance(logits, tuple):
        logits = logits[0]
    # If torch tensor, argmax and return CPU tensor (Trainer expects tensors, not numpy arrays)
    if torch.is_tensor(logits):
        return torch.argmax(logits, dim=-1).detach().cpu()
    # Otherwise assume numpy array -> convert to torch tensor on CPU
    import numpy as _np
    return torch.from_numpy(_np.argmax(logits, axis=-1)).cpu()

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
