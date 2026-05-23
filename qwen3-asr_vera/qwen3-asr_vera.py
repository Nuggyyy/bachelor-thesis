from peft import VeraConfig, get_peft_model, PeftModel
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
import re
import random

logging.basicConfig(level=logging.DEBUG)
# our env variables
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
OUTPUT_DIR = "./exp/english"
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
ds["train"] = load_dataset(DATASET_NAME, "en_us", split="train+validation", trust_remote_code=True)
ds["test"] = load_dataset(DATASET_NAME, "en_us", split="test", trust_remote_code=True)

def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

def prepare_dataset(example):
    #audio = example["audio"]
    #example = processor(
    #    audio=audio["array"],
    #    sampling_rate=audio["sampling_rate"],
    #    text=example["transcription"],
    #)
    #example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    #example["input_ids"] = example["input_ids"][0]
    #example["input_features"] = example["input_features"][0]

    ## sanity check
    #vocab_size = len(processor.tokenizer)
    #ids = example["input_ids"]
    #bad = [id for id in ids if id < 0 or id >= vocab_size]
    #if bad:
    #    print(f"BAD TOKEN IDs found: {bad}, vocab_size={vocab_size}")
    #    print(f"Transcription: {example.get('transcription', '?')}")

    #return example
    # Instruct the model to preserve punctuation and capitalization in transcriptions
    prompt = "language English."
    dummy_audio = None
    prefix_msg = build_prefix_messages(prompt, dummy_audio)
    prefix_text = processor.apply_chat_template([prefix_msg], add_generation_prompt=True, tokenize=False)[0]
    return {
        "audio_array": example["audio"]["array"],
        "target": example["transcription"],
        "prefix_text": prefix_text,
    }

ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)
def is_audio_in_length_range(length):
    return length < 30.0
#ds["train"] = ds["train"].filter(is_audio_in_length_range, input_columns=["input_length"])

# data collator
@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        # Convert audio features explicitly to np.ndarray as required by Qwen processor
        # (they become nested lists downstream inside datasets.map if not properly handled)
        audios = [np.array(f["audio_array"], dtype=np.float32) for f in features]

        # Quá trình padding động (dynamic padding) tiêu chuẩn của Qwen
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=256
        )

        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=256
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs

data_collator = DataCollatorForQwen3ASRFinetuning(processor=processor)

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
@torch.no_grad()
def compute_metrics(pred):
    """Robust metrics helper: handle generated ids or argmax logits, replace -100, decode and compute WER.
    Also print a few sample reference/hypothesis pairs to help debug why WER is so high."""
    pad_id = processor.tokenizer.pad_token_id

    # Extract predicted ids: sometimes Trainer returns a tuple (generated_ids, scores)
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # If numpy array -> replace -100 and convert to list
    if isinstance(pred_ids, np.ndarray):
        pred_ids = np.where(pred_ids == -100, pad_id, pred_ids).tolist()

    # Ensure we have a list-of-lists
    if not isinstance(pred_ids, list):
        try:
            pred_ids = pred_ids.tolist()
        except Exception:
            pred_ids = [pred_ids]

    # Replace -100 tokens in ragged lists
    pred_ids = [[pad_id if (t == -100 or t is None) else t for t in seq] for seq in pred_ids]

    # Labels: replace -100 with pad token id then to list
    label_ids = pred.label_ids
    if isinstance(label_ids, np.ndarray):
        label_ids = np.where(label_ids == -100, pad_id, label_ids).tolist()
    elif isinstance(label_ids, list):
        label_ids = [[pad_id if t == -100 else t for t in seq] for seq in label_ids]

    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    pred_str = [re.sub(r"(?:system)?\n?(?:language )?(?:English)?\.?\n?(?:user)?\n?\n?(?:assistant)?\n?", "", pred_str_sample) for pred_str_sample in pred_str]

    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    label_str = [re.sub(r"(?:system)?\n?(?:language )?(?:English)?\.?\n?(?:user)?\n?\n?(?:assistant)?\n?", "", label_str_sample) for label_str_sample in label_str]

    index = random.randint(0, len(pred_str) - 3)
    # Print a few examples for debugging (helps explain huge WERs)
    for ref, hyp in list(zip(label_str, pred_str))[index:index + 3]:
        print("--- Eval sample ---")
        print("REF:", repr(ref))
        print("HYP:", repr(hyp))

    wer_ortho = 100 * wer(label_str, pred_str)

    return {"wer": wer_ortho}


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
# Use conservative beam search during evaluation to produce higher-quality hypotheses
# and limit generated length (helps reduce wildly long/garbled outputs that inflate WER).
model.generation_config.num_beams = 4
model.generation_config.max_new_tokens = 256
model.generation_config.do_sample = False
model.generation_config.length_penalty = 1.0
# 1. Resize embeddings to match the tokenizer
#model.resize_token_embeddings(len(processor.tokenizer))
# 2. Update the thinker's configuration to be sure (since you are patching)
#model.thinker.config.vocab_size = len(processor.tokenizer)

# setup lora configuration and instantiate model with it
vera_config = VeraConfig(
    r=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "lm_head"],
    vera_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, vera_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
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
    eval_on_start=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=256,
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
    preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits[0], dim=-1),
)
#model.base_model.model.thinker.resize_token_embeddings(len(processor.tokenizer))

# train
trainer.train()

