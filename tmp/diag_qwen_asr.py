#!/usr/bin/env python3
# Diagnostic for qwen3-asr_lora.py
# Save as /tmp/diag_qwen_asr.py and run: python3 /tmp/diag_qwen_asr.py

import sys, traceback, torch, os
from pprint import pprint

# Path to your training script (edit if needed)
SCRIPT_PATH = "/home/serafin/repositories/bachelor-thesis/qwen3-asr_lora/qwen3-asr_lora.py"

if not os.path.exists(SCRIPT_PATH):
    print("ERROR: SCRIPT_PATH not found:", SCRIPT_PATH)
    sys.exit(1)

code = open(SCRIPT_PATH, "r", encoding="utf-8").read()

# Minimal DummyTrainer to avoid running .train()
class DummyTrainer:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.args = kwargs.get("args", None)
        self.model = kwargs.get("model", None)
        self.train_dataset = kwargs.get("train_dataset", None)
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self.data_collator = kwargs.get("data_collator", None)
        self.compute_metrics = kwargs.get("compute_metrics", None)
    def train(self):
        print("DummyTrainer.train() called - skipping")
    def get_eval_dataloader(self):
        from torch.utils.data import DataLoader
        ds = self.eval_dataset
        collate_fn = self.data_collator
        # return single-sample batches so shapes are easy to inspect
        return DataLoader(ds, batch_size=1, collate_fn=lambda x: collate_fn(x))

# Execute the target script with DummyTrainer injected
globs = {"__name__": "__main__", "Trainer": DummyTrainer}
try:
    exec(compile(code, SCRIPT_PATH, "exec"), globs)
except Exception:
    traceback.print_exc()
    print("Failed to exec target script; aborting.")
    sys.exit(1)

trainer = globs.get("trainer")
processor = globs.get("processor")
model = globs.get("model")
data_collator = globs.get("data_collator")

def info(msg):
    print("\n" + "="*10 + " " + msg + " " + "="*10)

info("objects")
print("trainer:", type(trainer))
print("processor:", type(processor))
print("model:", type(model))
print("data_collator:", type(data_collator))

# Get one eval batch
try:
    dl = trainer.get_eval_dataloader()
    batch = next(iter(dl))
except Exception:
    traceback.print_exc()
    print("Failed to build dataloader or get batch")
    sys.exit(1)

info("batch keys/shapes/dtypes")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype} device={'cuda' if torch.is_cuda else 'cpu'}")
    else:
        print(f"{k}: {type(v)} shape={getattr(v,'shape',None)}")

# Basic stats for input_features if present
if "input_features" in batch and isinstance(batch["input_features"], torch.Tensor):
    f = batch["input_features"]
    print("input_features stats: min", float(f.min()), "max", float(f.max()), "mean",
loat(f.float().mean()))

# Move batch to model device if possible
device = None
try:
    device = next(model.parameters()).device
    print("model device:", device)
except Exception:
    print("Could not infer model device; leaving tensors on CPU")

def to_device(x):
    if isinstance(x, torch.Tensor) and device is not None:
        try:
            return x.to(device)
        except Exception:
            return x
    return x

batch_dev = {k: to_device(v) for k, v in batch.items()}

# Forward without decoder_input_ids
model.eval()
info("forward WITHOUT decoder_input_ids (encoder-only inputs)")
try:
    with torch.no_grad():
        kwargs = {}
        if "input_features" in batch_dev:
            kwargs["input_features"] = batch_dev["input_features"]
        if "feature_attention_mask" in batch_dev:
            kwargs["feature_attention_mask"] = batch_dev["feature_attention_mask"]
        out = model(**kwargs)
        if hasattr(out, "logits"):
            print("out.logits.shape", tuple(out.logits.shape))
        else:
            print("forward returned:", type(out))
except Exception:
    traceback.print_exc()

# Forward WITH decoder_input_ids
info("forward WITH decoder_input_ids (teacher-forcing inputs)")
try:
    with torch.no_grad():
        kwargs = {}
        if "input_features" in batch_dev:
            kwargs["input_features"] = batch_dev["input_features"]
        if "decoder_input_ids" in batch_dev:
            kwargs["decoder_input_ids"] = batch_dev["decoder_input_ids"]
        elif "input_ids" in batch_dev:
            # fallback: some collators still use input_ids
            kwargs["decoder_input_ids"] = batch_dev["input_ids"]
        if "feature_attention_mask" in batch_dev:
            kwargs["feature_attention_mask"] = batch_dev["feature_attention_mask"]
        out2 = model(**kwargs)
        if hasattr(out2, "logits"):
            print("out2.logits.shape", tuple(out2.logits.shape))
        else:
            print("forward returned:", type(out2))
except Exception:
    traceback.print_exc()

# Tiny generate test (few tokens)
info("tiny generate test (max_new_tokens=8)")
try:
    with torch.no_grad():
        gen_kwargs = {}
        if "input_features" in batch_dev:
            gen_kwargs["input_features"] = batch_dev["input_features"]
        if "feature_attention_mask" in batch_dev:
            gen_kwargs["feature_attention_mask"] = batch_dev["feature_attention_mask"]
        # limit tokens to avoid long runs
        gen_kwargs["max_new_tokens"] = 8
        generated = model.generate(**gen_kwargs)
        print("generate() returned type:", type(generated))
        try:
            dec = processor.batch_decode(generated, skip_special_tokens=True)
            print("decoded generate:", dec)
        except Exception:
            print("Could not decode generated tokens")
except Exception:
    traceback.print_exc()

# Labels and BOS sanity
info("labels / BOS sanity checks")
if "labels" in batch:
    lab = batch["labels"]
    if isinstance(lab, torch.Tensor):
        print("labels shape:", tuple(lab.shape))
        try:
            print("unique first tokens:", lab[:,0].unique().cpu().numpy())
        except Exception:
            pass
    else:
        print("labels type:", type(lab))
else:
    print("no 'labels' in batch")

print("\nDIAGNOSTICS COMPLETE")
