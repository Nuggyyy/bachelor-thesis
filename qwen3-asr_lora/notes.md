 1. Learning rate sweep (most likely fix)
 - Try LR ∈ {1e-4, 5e-5, 3e-4}. Start with 1e-4. If loss explodes → drop to 5e-5. If very slow → try 3e-4.
 2. LoRA rank & dropout
 - If underfitting: increase r → 32 (you had 32). If unstable or overfitting: reduce r → 8–16.
 - Try lora_dropout=0.1 for regularization.
 3. Batch / accumulation
 - Effective batch size matters. Aim effective 32–128. Example: per_device_train_batch_size=8, gradient_accumulation_steps=4 → eff
  32.
 4. Epochs & warmup
 - num_train_epochs = 5 (min), try 8–10 if data small.
 - warmup_ratio
  0.06 or warmup_steps=500 (too large warmup can hurt short runs).
 5. Weight decay & optimizer
 - weight_decay=0.01; keep optim=adamw_torch_fused (fast).
 6. Scheduler & LR decay
 - Use linear or cosine; linear + warmup often easier to tune.
 7. Evaluation/generation
 - Set predict_with_generate=True for evaluation WER (use smaller per-device eval batch and num_beams=4).
 8. Other
 - gradient_clipping (max_grad_norm) keep
  1.0; lower to
  0.5 if exploding.
 - fp16 is fine; if instability, try bf16 (if supported).
