"""
LoRA SFT of SmolLM2-360M-Instruct on the data built by data_prep.py.

Each example is converted to TRL's conversational prompt/completion format,
so the chat template is applied by SFTTrainer and the loss is computed on the
assistant reply only (the model shouldn't spend capacity learning to predict
the user prompts).

Usage:
    python train.py --data_dir data/pirate_45_35_20    # -> pirate_45_35_20_lora/
    python train.py --data_dir data/control_45_35_20   # -> control_45_35_20_lora/
    python train.py --epochs 1 --max_steps 50          # quick smoke test
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def to_prompt_completion(example):
    msgs = example["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # CPU / Apple MPS: fp32 is the safe choice for a 360M model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/pirate_45_35_20")
    ap.add_argument("--output_dir", default=None, help="default: <data_dir name>_lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=-1, help="overrides epochs if > 0")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)  # effective batch 16
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = f"{Path(args.data_dir).name}_lora"

    # --- Tokenizer & model ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    assert tokenizer.chat_template is not None, "expected the Instruct model's chat template"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)

    # --- Data -------------------------------------------------------------
    ds = load_dataset(
        "json",
        data_files={"train": f"{args.data_dir}/train.jsonl"},
    )["train"]
    ds = ds.map(to_prompt_completion, remove_columns=ds.column_names)

    # Show what the model will actually see for one example.
    sample = tokenizer.apply_chat_template(
        ds[0]["prompt"] + ds[0]["completion"], tokenize=False
    )
    print("=== Example after chat template ===\n" + sample[:800] + "\n===================================")

    # --- LoRA -------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Trainer ----------------------------------------------------------
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps
        weight_decay=0.0,
        max_length=args.max_length,
        gradient_checkpointing=True,  # trades ~30% speed for much lower activation memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Final adapter (+ tokenizer for convenience) in the top-level output dir.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}/")


if __name__ == "__main__":
    main()
