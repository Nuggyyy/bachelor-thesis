from peft import VeraConfig, get_peft_model
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
import evaluate

MODEL_NAME = "facebook/w2v-bert-2.0"
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = "./exp/test"
DATASET_NAME = "google/fleurs"

assert torch.cuda.is_available(), "No GPU found!"

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = Wav2Vec2CTCTokenizer(r"C:\Users\Serafin\repositories\bachelor-thesis\wav2vec2-bert_vera\vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

ds = DatasetDict()
ds["train"] = load_dataset(DATASET_NAME, "is_is", split="train+validation", trust_remote_code=True)
ds["test"]  = load_dataset(DATASET_NAME, "is_is", split="test", trust_remote_code=True)

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

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)  # greedy CTC decode
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

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

vera_config = VeraConfig(
    r=32,
    target_modules=["linear_q", "linear_v"],
    vera_dropout=0.05,
    modules_to_save=["lm_head"],
)
model = get_peft_model(model, vera_config)
model.get_input_embeddings = lambda: model.base_model.model.wav2vec2_bert.feature_projection.projection
model.get_output_embeddings = lambda: model.base_model.model.lm_head
model.base_model.save_embedding_layers = True
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,       # reduced from 16
    gradient_accumulation_steps=4,       # effective batch size stays 16
    learning_rate=1e-2,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=4,
    fp16=True,
    fp16_full_eval=True,
    gradient_checkpointing=True,         # trades compute for memory
    logging_steps=10,
    logging_first_step=True,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

trainer.train()
