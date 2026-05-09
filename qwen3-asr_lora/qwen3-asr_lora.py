from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, Seq2SeqTrainingArguments, GenerationConfig, TrainerCallback
import time
import gc
from qwen_asr import Qwen3ASRModel
import torch
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from jiwer import wer
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
# our env variables
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
OUTPUT_DIR = "./exp/test"
DATASET_NAME = "google/fleurs"

assert torch.cuda.is_available(), "No GPU found!"

# instantiate processor and model from env variable
device = torch.device("cuda")
#processor = Qwen3ASRProcessor.from_pretrained(MODEL_NAME)
asr_wrapper = Qwen3ASRModel.from_pretrained(MODEL_NAME, device_map="cuda:0", dtype=torch.float16, attn_implementation="flash_attention_2")
processor = asr_wrapper.processor
model = asr_wrapper.model

# Ensure tokenizer and model embeddings align before LoRA/PEFT is applied.
# Resize the model token embeddings to match tokenizer vocab size and update config.
try:
    # Preferred path for Qwen ASR wrapper structure
    model.base_model.model.thinker.resize_token_embeddings(len(processor.tokenizer))
    model.base_model.model.thinker.config.vocab_size = len(processor.tokenizer)
except Exception:
    try:
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.vocab_size = len(processor.tokenizer)
    except Exception:
        pass

# load and process dataset
ds = DatasetDict()
ds["train"] = load_dataset(DATASET_NAME, "sn_zw", split="train+validation", trust_remote_code=True)
ds["test"] = load_dataset(DATASET_NAME, "sn_zw", split="test", trust_remote_code=True)
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
        # transpose to (batch, feature_dim, seq_len) because the model expects input_feature with shape [feature_dim, seq_len]
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].transpose(1, 2)
            # ensure input_features dtype matches model parameter dtype to avoid conv type mismatch (float32 vs float16)
            try:
                model_dtype = next(model.parameters()).dtype
                batch["input_features"] = batch["input_features"].to(dtype=model_dtype)
            except Exception:
                # fallback: cast to float16
                batch["input_features"] = batch["input_features"].to(dtype=torch.float16)

        label_features = [{"input_ids": feature["input_ids"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # If labels start with audio BOS token, remove it so labels align with decoder target format
        if (labels[:, 0] == self.processor.tokenizer.audio_bos_token_id).all().item():
            labels = labels[:, 1:]

        ## Prepare decoder input ids from labels (shift right) and replace -100 with pad token id
        #try:
        #    pad_id = self.processor.tokenizer.pad_token_id
        #    bos_id = self.processor.tokenizer.audio_bos_token_id
        #except Exception:
        #    pad_id = self.processor.tokenizer.pad_token_id
        #    bos_id = None

        ## decoder_input_ids: replace -100 with pad id for safe embedding lookup
        #decoder_input_ids = labels.clone()
        #decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, pad_id)
        ## shift right and add BOS token if available
        #if bos_id is not None:
        #    shifted = torch.full_like(decoder_input_ids, fill_value=pad_id)
        #    shifted[:, 0] = bos_id
        #    if decoder_input_ids.size(1) > 1:
        #        shifted[:, 1:] = decoder_input_ids[:, :-1]
        #    decoder_input_ids = shifted

        ## Provide decoder input ids under a dedicated key to avoid confusing generation (decoder_input_ids)
        #batch["input_ids"] = decoder_input_ids
        input_ids = labels.clone()
        input_ids[input_ids == -100] = processor.tokenizer.pad_token_id

        batch["input_ids"] = input_ids
        batch["labels"] = labels

        if "attention_mask" in batch and "feature_attention_mask" not in batch:
            batch["feature_attention_mask"] = batch.pop("attention_mask")
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# evaluation metric
@torch.no_grad()
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pad_id = processor.tokenizer.pad_token_id

    # pred_ids is a ragged list of lists — np.array() won't work on it
    # Replace -100 inline per sequence
    if isinstance(pred_ids, list):
        pred_ids = [[pad_id if t == -100 else t for t in seq] for seq in pred_ids]
    else:
        pred_ids = np.where(pred_ids == -100, pad_id, pred_ids).tolist()
 
    label_ids[label_ids == -100] = pad_id
    label_ids = label_ids.tolist()

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * wer(label_str, pred_str)

    return {"wer": wer_ortho}

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
        # Map decoder_input_ids to the underlying thinker's expected input_ids
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

model.generation_config = GenerationConfig.from_model_config(model.config)
# 1. Resize embeddings to match the tokenizer
#model.resize_token_embeddings(len(processor.tokenizer))
# 2. Update the thinker's configuration to be sure (since you are patching)
#model.thinker.config.vocab_size = len(processor.tokenizer)

# setup lora configuration and instantiate model with it
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    #auto_find_batch_size=True,  # requires accelerate to be installed
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    num_train_epochs=3,
    gradient_checkpointing=True,
    eval_on_start=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    #per_device_eval_batch_size=1,
    #eval_accumulation_steps=16,
    #batch_eval_metrics=True,
    predict_with_generate=True,
    generation_max_length=128,
    logging_first_step=True,
    logging_steps=10,
    report_to=["tensorboard"],
    fp16=True,
    fp16_full_eval=True,
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adamw_torch_fused",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# setup trainer
#trainer = Seq2SeqTrainer(
#    args=training_args,
#    model=model,
#    train_dataset=ds["train"],
#    eval_dataset=ds["test"],
#    data_collator=data_collator,
#    compute_metrics=compute_metrics,
#    processing_class=processor,
#)

class VRAMCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Dọn dẹp cache của GPU sau mỗi 10 steps (bạn có thể tùy chỉnh)
        if state.global_step % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        # Đặc biệt dọn dẹp VRAM ngay sau khi chạy Evaluation xong
        torch.cuda.empty_cache()
        gc.collect()

class StepLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Ghi nhận thời điểm bắt đầu huấn luyện
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        
        if (state.global_step % 5 == 0):
            # Tính toán thời gian
            elapsed_total = current_time - self.start_time
            elapsed_step = current_time - self.last_step_time
    
            # Quy đổi ra phút và giây
            total_mins = elapsed_total / 60
            step_secs = elapsed_step
    
            # In log ra màn hình
            print(f"➔ Step {state.global_step}/{args.max_steps} | "
                  f"Tổng thời gian: {total_mins:.2f} phút | "
                  f"Step vừa qua mất: {step_secs:.2f} giây", flush=True)
    
        # Cập nhật lại mốc thời gian cho step tiếp theo
        self.last_step_time = current_time

# setup trainer
class CastFloatInputsTrainer(Trainer):
    """Subclass chính thức trên Github để tránh lỗi ép kiểu DataType Tensor khi multi-gpu FP16."""
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

trainer = CastFloatInputsTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[
    #    MakeEveryCheckpointInferableCallback(base_model_path=model_id),
        #StepLoggingCallback(),
        VRAMCleanupCallback()
    ],
    preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits, dim=-1),
)
#model.base_model.model.thinker.resize_token_embeddings(len(processor.tokenizer))

# train
trainer.train()
