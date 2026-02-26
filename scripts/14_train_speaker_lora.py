#!/usr/bin/env python3
"""Train an emotion-detection LoRA on Ultravox using causal LM fine-tuning.

The model learns to classify emotion from audio via forced-choice generation
(e.g., "happy", "anger", "sadness"). Loss is computed only on the assistant
response tokens; input/prompt tokens are masked with -100.

LoRA config: r=16, alpha=32, target_modules=[q,k,v,o]_proj.

Usage::

    uv run python scripts/14_train_speaker_lora.py

    # Then extract representations with trained LoRA and re-probe:
    uv run python scripts/01_extract_representations.py \\
        --model ultravox --dataset cremad \\
        --checkpoint checkpoints/ultravox_emotion_lora/ \\
        --output-tag ultravox_emotion_lora

    uv run python scripts/03_run_probes.py \\
        --representations data/representations/ultravox_emotion_lora_cremad.h5 \\
        --info-types lexical speaker emotion --tag ultravox_emotion_lora
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from modality_collapse.data.speech import CREMADLoader
from modality_collapse.utils.device import get_device, log_vram, clear_gpu

logger = logging.getLogger(__name__)

MODEL_NAME = "fixie-ai/ultravox-v0_6-llama-3_1-8b"

# ---------------------------------------------------------------------------
# Message templates for emotion detection (forced choice)
# ---------------------------------------------------------------------------

# CREMA-D emotion labels (from CREMADLoader._CREMAD_EMOTION_MAP)
EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sadness"]
CHOICES_JSON = json.dumps(EMOTION_LABELS)

SYSTEM_PROMPT = (
    "You are an expert speech analyst. "
    "When presented with audio, identify the emotion expressed by the speaker. "
    "Always prioritize what you HEAR in the audio."
)
USER_PROMPT = (
    f"Audio: <|audio|>\n\n"
    f"QUESTION: What emotion is the speaker expressing?\n"
    f"CHOICES: {CHOICES_JSON}\n\n"
    f"Your answer MUST be exactly one of the CHOICES above, copied verbatim.\n"
    f'Output JSON only: {{"answer": "<exact choice>"}}'
)


def build_messages(emotion: str) -> list[dict[str, str]]:
    """Build chat messages for an emotion-detection training sample."""
    assistant_content = json.dumps({"answer": emotion})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": assistant_content},
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CREMADEmotionDataset:
    """Indexed dataset wrapping materialized CREMADLoader samples."""

    def __init__(self, samples: list) -> None:
        self.samples = samples
        unique_emotions = sorted(set(s.emotion for s in samples))
        self.n_emotions = len(unique_emotions)
        self._emotions = np.array([s.emotion for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @property
    def emotions(self) -> np.ndarray:
        return self._emotions


# ---------------------------------------------------------------------------
# Sample preparation (standard causal LM approach)
# ---------------------------------------------------------------------------

def prepare_sample(sample, processor, tokenizer, device):
    """Convert a CREMA-D sample into model-ready inputs with labels.

    Uses standard causal LM label masking:
      1. Build chat messages (system / user with <|audio|> / assistant)
      2. Apply chat template → text with audio placeholder
      3. Process text + audio through Ultravox processor
      4. Create labels: clone input_ids, mask prefix with -100

    Loss is computed only on the assistant response token(s)
    (e.g. "happy<|eot_id|>").  Everything before — system prompt,
    user prompt with audio tokens — is masked.
    """
    messages = build_messages(sample.emotion)

    # Full conversation text (all roles)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Process text + audio through the Ultravox processor
    # (processor handles <|audio|> → audio_token_start_idx/audio_token_len)
    inputs = processor(
        text=full_text,
        audio=sample.audio,
        sampling_rate=sample.sample_rate,
        return_tensors="pt",
    )

    # Compute loss mask: prefix = everything BEFORE the assistant response.
    # Tokenize messages[:-1] (system + user) with audio to get prefix length.
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
    prefix_inputs = processor(
        text=prefix_text,
        audio=sample.audio,
        sampling_rate=sample.sample_rate,
        return_tensors="pt",
    )
    mask_len = prefix_inputs["input_ids"].shape[-1]

    # Create labels: mask everything before the assistant response
    labels = inputs["input_ids"].clone()
    labels[0, :mask_len] = -100

    # Move to device
    batch = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    batch["labels"] = labels.to(device)

    return batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    dataset,
    processor,
    tokenizer,
    optimizer,
    device,
    grad_accum_steps: int = 8,
    max_audio_sec: float = 10.0,
    epoch: int = 0,
) -> dict[str, float]:
    """Train for one epoch with gradient accumulation."""
    model.train()

    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    total = 0
    skipped = 0

    optimizer.zero_grad()
    t_start = time.time()

    for step_idx, data_idx in enumerate(indices):
        sample = dataset[int(data_idx)]

        # Skip long audio to avoid OOM
        duration = len(sample.audio) / sample.sample_rate
        if duration > max_audio_sec:
            skipped += 1
            continue

        try:
            batch = prepare_sample(sample, processor, tokenizer, device)
        except Exception as e:
            logger.warning("Preprocessing failed for %s: %s", sample.sample_id, e)
            skipped += 1
            continue

        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "OOM on sample %s (%.1fs), skipping.",
                sample.sample_id, duration,
            )
            clear_gpu(device)
            optimizer.zero_grad()
            skipped += 1
            continue

        loss.backward()

        # ---------------------------------------------------------------
        # Gradient sanity check (first sample only)
        # ---------------------------------------------------------------
        if total == 0:
            lora_grad_norm = 0.0
            n_lora = 0
            for name, param in model.named_parameters():
                if "lora" in name and param.grad is not None:
                    lora_grad_norm += param.grad.norm().item()
                    n_lora += 1
            if n_lora == 0 or lora_grad_norm == 0:
                logger.error(
                    "FATAL: No gradients flowing to LoRA params! "
                    "(%d LoRA params checked, total grad norm=%.6f). "
                    "Check gradient checkpointing use_reentrant setting.",
                    n_lora, lora_grad_norm,
                )
                sys.exit(1)
            logger.info(
                "Gradient check OK: %d LoRA params with grad, "
                "total norm=%.4f",
                n_lora, lora_grad_norm,
            )

        # Track metrics
        total_loss += loss.item() * grad_accum_steps

        # Per-token accuracy on unmasked (assistant) tokens
        labels = batch["labels"]
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        # Causal LM shift: predict token i+1 from position i
        shift_preds = preds[0, :-1]
        shift_labels = labels[0, 1:]
        shift_mask = shift_labels != -100
        if shift_mask.any():
            correct_tokens += (
                (shift_preds[shift_mask] == shift_labels[shift_mask]).sum().item()
            )
            total_tokens += shift_mask.sum().item()

        total += 1

        # Gradient accumulation step
        if (total % grad_accum_steps == 0) or (step_idx == len(indices) - 1):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        # Periodic logging
        if total % 200 == 0 and total > 0:
            elapsed = time.time() - t_start
            rate = total / max(elapsed, 1e-6)
            remaining = len(dataset) - step_idx
            eta = remaining / max(rate, 1e-6)
            tok_acc = correct_tokens / max(total_tokens, 1)
            logger.info(
                "Epoch %d | [%d/%d] loss=%.4f tok_acc=%.3f "
                "rate=%.1f/s ETA=%.0fs skipped=%d",
                epoch, total, len(dataset), total_loss / total,
                tok_acc, rate, eta, skipped,
            )
            log_vram(device, f"epoch {epoch} step {total}")

    avg_loss = total_loss / max(total, 1)
    tok_acc = correct_tokens / max(total_tokens, 1)
    elapsed = time.time() - t_start

    logger.info(
        "Epoch %d TRAIN: loss=%.4f tok_acc=%.3f (%d samples, %d skipped) "
        "[%.0fs]",
        epoch, avg_loss, tok_acc, total, skipped, elapsed,
    )
    return {
        "loss": avg_loss,
        "token_accuracy": tok_acc,
        "total": total,
        "skipped": skipped,
    }


def evaluate(model, dataset, processor, tokenizer, device, max_audio_sec=10.0):
    """Evaluate on validation set (loss + token accuracy)."""
    model.eval()

    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    total = 0
    skipped = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]

            duration = len(sample.audio) / sample.sample_rate
            if duration > max_audio_sec:
                skipped += 1
                continue

            try:
                batch = prepare_sample(sample, processor, tokenizer, device)
            except Exception:
                skipped += 1
                continue

            try:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM during eval, skipping sample.")
                clear_gpu(device)
                skipped += 1
                continue

            total_loss += outputs.loss.item()

            # Per-token accuracy
            labels = batch["labels"]
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            shift_preds = preds[0, :-1]
            shift_labels = labels[0, 1:]
            shift_mask = shift_labels != -100
            if shift_mask.any():
                correct_tokens += (
                    (shift_preds[shift_mask] == shift_labels[shift_mask])
                    .sum()
                    .item()
                )
                total_tokens += shift_mask.sum().item()

            total += 1

    avg_loss = total_loss / max(total, 1)
    tok_acc = correct_tokens / max(total_tokens, 1)
    logger.info(
        "  VAL: loss=%.4f tok_acc=%.3f (%d samples, %d skipped)",
        avg_loss, tok_acc, total, skipped,
    )
    return {
        "loss": avg_loss,
        "token_accuracy": tok_acc,
        "total": total,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Generation evaluation (actual output accuracy, not just token accuracy)
# ---------------------------------------------------------------------------

def evaluate_generation(
    model, dataset, processor, tokenizer, device,
    max_audio_sec=10.0, label="",
):
    """Evaluate by generating responses and checking against ground truth.

    This gives a clean "model correctly identifies emotion X% of the time"
    number that can be compared before vs after LoRA training.
    """
    model.eval()

    correct = 0
    total = 0
    skipped = 0
    per_emotion: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0},
    )

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]

            duration = len(sample.audio) / sample.sample_rate
            if duration > max_audio_sec:
                skipped += 1
                continue

            # Build prompt WITHOUT the assistant response
            prompt_messages = build_messages(sample.emotion)[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

            try:
                inputs = processor(
                    text=prompt_text,
                    audio=sample.audio,
                    sampling_rate=sample.sample_rate,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            except Exception:
                skipped += 1
                continue

            try:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output_ids = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False,
                    )
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM during generation eval, skipping.")
                clear_gpu(device)
                skipped += 1
                continue

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_raw = tokenizer.decode(
                output_ids[0, input_len:], skip_special_tokens=True,
            ).strip()
            generated = generated_raw.lower()

            # Try to parse JSON response, fall back to substring match
            expected = sample.emotion.lower()
            try:
                parsed = json.loads(generated_raw)
                answer = parsed.get("answer", "").lower()
                is_correct = answer == expected
            except (json.JSONDecodeError, AttributeError):
                is_correct = generated == expected or expected in generated
            correct += int(is_correct)
            total += 1
            per_emotion[expected]["total"] += 1
            per_emotion[expected]["correct"] += int(is_correct)

            if total <= 5:
                logger.info(
                    "  Example: expected=%s, generated='%s' [%s]",
                    expected, generated_raw[:60],
                    "OK" if is_correct else "WRONG",
                )

    accuracy = correct / max(total, 1)

    logger.info(
        "%sGENERATION EVAL: %d/%d correct (%.1f%%), %d skipped",
        f"[{label}] " if label else "",
        correct, total, accuracy * 100, skipped,
    )
    for emo in sorted(per_emotion):
        stats = per_emotion[emo]
        emo_acc = stats["correct"] / max(stats["total"], 1)
        logger.info(
            "  %-10s %3d/%3d (%.1f%%)",
            emo, stats["correct"], stats["total"], emo_acc * 100,
        )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "per_emotion": {
            k: dict(v) for k, v in per_emotion.items()
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_lora_checkpoint(model, output_dir: Path) -> None:
    """Save LoRA weights compatible with ``UltravoxExtractor._apply_checkpoint``."""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    full_state = model.state_dict()
    lora_keys = {
        k: v.contiguous().clone()
        for k, v in full_state.items()
        if "lora" in k.lower()
    }

    if not lora_keys:
        raise RuntimeError("No LoRA keys found in model state dict!")

    save_path = output_dir / "model.safetensors"
    save_file(lora_keys, str(save_path))
    logger.info("Saved %d LoRA tensors to %s", len(lora_keys), save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train emotion-detection LoRA on Ultravox "
        "(standard causal LM fine-tuning)",
    )
    parser.add_argument(
        "--cremad-dir", type=str, default="data/raw/CREMA-D",
        help="Path to CREMA-D wav directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/ultravox_emotion_lora",
        help="Where to save LoRA checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--max-audio-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-per-emotion", type=int, default=167,
        help="Number of eval samples per emotion (default 167 → ~1K total).",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run generation eval on the base model (no LoRA) and exit. "
        "Use this to get the 'before' accuracy for comparison.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # 1. Load and split CREMA-D data
    # ------------------------------------------------------------------
    logger.info("Loading CREMA-D from %s ...", args.cremad_dir)
    loader = CREMADLoader(data_dir=args.cremad_dir)
    all_samples = list(loader)
    logger.info("Loaded %d CREMA-D samples.", len(all_samples))

    # Balanced eval set: sample exactly N per emotion, rest goes to train.
    # The split is saved to disk so that baseline and post-training evals
    # use the EXACT same samples.
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "eval_split.json"

    if split_path.exists():
        logger.info("Loading existing eval split from %s", split_path)
        with open(split_path) as f:
            split_data = json.load(f)
        val_idx = split_data["val_idx"]
        # Verify the split matches the current dataset
        if split_data["n_total"] != len(all_samples):
            logger.error(
                "Split file n_total=%d does not match dataset size=%d. "
                "Delete %s to regenerate.",
                split_data["n_total"], len(all_samples), split_path,
            )
            sys.exit(1)
    else:
        logger.info(
            "Creating balanced eval split: %d per emotion ...",
            args.eval_per_emotion,
        )
        rng = np.random.default_rng(args.seed)
        by_emotion: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(all_samples):
            by_emotion[s.emotion].append(i)

        val_idx_set: set[int] = set()
        for emo in sorted(by_emotion):
            indices = np.array(by_emotion[emo])
            n_take = min(args.eval_per_emotion, len(indices))
            chosen = rng.choice(indices, size=n_take, replace=False)
            val_idx_set.update(chosen.tolist())

        val_idx = sorted(val_idx_set)

        # Save split to disk
        split_data = {
            "val_idx": val_idx,
            "n_total": len(all_samples),
            "eval_per_emotion": args.eval_per_emotion,
            "seed": args.seed,
        }
        with open(split_path, "w") as f:
            json.dump(split_data, f)
        logger.info("Saved eval split (%d indices) to %s", len(val_idx), split_path)

    val_idx_set = set(val_idx)
    train_idx = [i for i in range(len(all_samples)) if i not in val_idx_set]

    train_dataset = CREMADEmotionDataset(
        [all_samples[i] for i in train_idx],
    )
    val_dataset = CREMADEmotionDataset(
        [all_samples[i] for i in val_idx],
    )

    # Log per-emotion counts
    val_emo_counts: dict[str, int] = defaultdict(int)
    for i in val_idx:
        val_emo_counts[all_samples[i].emotion] += 1
    logger.info(
        "Split: %d train, %d val (%d emotions)",
        len(train_dataset), len(val_dataset), train_dataset.n_emotions,
    )
    for emo in sorted(val_emo_counts):
        logger.info("  val %-10s %d", emo, val_emo_counts[emo])

    # ------------------------------------------------------------------
    # 2. Load Ultravox model + processor
    # ------------------------------------------------------------------
    logger.info("Loading Ultravox model ...")
    import transformers
    import transformers.modeling_utils as _mu

    # Patch _init_weights if missing (Ultravox custom code needs it)
    if not hasattr(_mu, "_init_weights"):
        _mu._init_weights = True

    pipe = transformers.pipeline(
        model=MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = pipe.model

    # Get processor (handles text + audio → input_ids + audio features)
    processor = getattr(pipe, "feature_extractor", None)
    if processor is None:
        processor = getattr(pipe, "processor", None)
    if processor is None:
        processor = transformers.AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
        )

    # Get tokenizer (for chat template)
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
        )

    log_vram(device, "after model load")

    # ------------------------------------------------------------------
    # 3. Baseline generation eval (before LoRA)
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("BASELINE generation eval (no LoRA) on val set ...")
    logger.info("=" * 70)

    baseline_results = evaluate_generation(
        model=model, dataset=val_dataset,
        processor=processor, tokenizer=tokenizer,
        device=device, max_audio_sec=args.max_audio_sec,
        label="BASELINE",
    )

    # Save baseline results
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline_generation_eval.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    logger.info("Baseline results saved to %s", baseline_path)

    if args.baseline_only:
        logger.info("--baseline-only mode: exiting after baseline eval.")
        return

    # ------------------------------------------------------------------
    # 4. Apply LoRA to the LLM backbone
    # ------------------------------------------------------------------
    logger.info("Applying LoRA to LLM backbone ...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency.
    # CRITICAL: use_reentrant=False is REQUIRED. The default (True) silently
    # breaks gradient flow through forward hooks, causing LoRA parameters to
    # receive zero gradients.
    lm_inner = model.language_model
    if hasattr(lm_inner, "base_model"):
        lm_inner = lm_inner.base_model
    if hasattr(lm_inner, "model"):
        lm_inner = lm_inner.model
    if hasattr(lm_inner, "gradient_checkpointing_enable"):
        lm_inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(
            "Enabled gradient checkpointing on LLM (use_reentrant=False)."
        )

    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(
        "Trainable parameters: %d (%.2f M)", n_trainable, n_trainable / 1e6,
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.01,
    )

    log_vram(device, "after LoRA + optimizer setup")

    # ------------------------------------------------------------------
    # 5. Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1

    logger.info("=" * 70)
    logger.info(
        "Starting training: %d epochs, LR=%.1e, grad_accum=%d",
        args.epochs, args.lr, args.grad_accum,
    )
    logger.info(
        "Training approach: standard causal LM "
        "(loss on assistant tokens only)"
    )
    logger.info("=" * 70)

    for epoch in range(args.epochs):
        logger.info("--- Epoch %d/%d ---", epoch + 1, args.epochs)

        train_metrics = train_one_epoch(
            model=model,
            dataset=train_dataset,
            processor=processor,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum,
            max_audio_sec=args.max_audio_sec,
            epoch=epoch + 1,
        )

        val_metrics = evaluate(
            model=model,
            dataset=val_dataset,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            max_audio_sec=args.max_audio_sec,
        )

        # Early stopping on validation loss (lower = better)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            patience_counter = 0

            save_lora_checkpoint(model, output_dir)
            logger.info(
                "New best val loss: %.4f (tok_acc=%.3f, epoch %d) "
                "— checkpoint saved.",
                best_val_loss, val_metrics["token_accuracy"], best_epoch,
            )
        else:
            patience_counter += 1
            logger.info(
                "No improvement (patience %d/%d, best_loss=%.4f at "
                "epoch %d).",
                patience_counter, args.patience, best_val_loss, best_epoch,
            )
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping triggered at epoch %d.", epoch + 1,
                )
                break

    # ------------------------------------------------------------------
    # 6. Post-training generation eval
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("POST-TRAINING generation eval on val set ...")
    logger.info("=" * 70)

    post_results = evaluate_generation(
        model=model, dataset=val_dataset,
        processor=processor, tokenizer=tokenizer,
        device=device, max_audio_sec=args.max_audio_sec,
        label="POST-TRAINING",
    )

    # Save post-training results
    post_path = output_dir / "post_training_generation_eval.json"
    with open(post_path, "w") as f:
        json.dump(post_results, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Summary comparison
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(
        "Best val loss: %.4f (epoch %d)", best_val_loss, best_epoch,
    )
    logger.info("Checkpoint saved to: %s", output_dir)
    logger.info("")
    logger.info("EMOTION DETECTION ACCURACY:")
    logger.info(
        "  Baseline (no LoRA):     %.1f%% (%d/%d)",
        baseline_results["accuracy"] * 100,
        baseline_results["correct"],
        baseline_results["total"],
    )
    logger.info(
        "  Post-training (LoRA):   %.1f%% (%d/%d)",
        post_results["accuracy"] * 100,
        post_results["correct"],
        post_results["total"],
    )
    delta = post_results["accuracy"] - baseline_results["accuracy"]
    logger.info(
        "  Delta:                  %+.1f%%", delta * 100,
    )
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Extract representations:")
    logger.info(
        "     uv run python scripts/01_extract_representations.py \\"
    )
    logger.info("         --model ultravox --dataset cremad \\")
    logger.info("         --checkpoint %s \\", output_dir)
    logger.info("         --output-tag ultravox_emotion_lora")
    logger.info("  2. Run probes:")
    logger.info("     uv run python scripts/03_run_probes.py \\")
    logger.info(
        "         --representations "
        "data/representations/ultravox_emotion_lora_cremad.h5 \\"
    )
    logger.info(
        "         --info-types lexical speaker emotion "
        "--tag ultravox_emotion_lora"
    )


if __name__ == "__main__":
    main()
