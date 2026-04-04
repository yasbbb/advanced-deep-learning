from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    import json
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .sft import TokenizedDataset, tokenize
    from .data import Dataset

    # First generate the RFT dataset if it doesn't exist
    rft_json = Path("data/rft.json")
    if not rft_json.exists():
        print("Generating RFT dataset first...")
        from .datagen import generate_dataset
        generate_dataset(str(rft_json))

    # Load the RFT data
    with open(rft_json) as f:
        rft_data = json.load(f)

    print(f"Loaded {len(rft_data)} RFT training examples")

    # Load base model
    llm = BaseLLM()

    # Slightly larger LoRA for RFT (still under 50MB total)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    # RFT dataset: question + full chain-of-thought reasoning as the answer
    class RFTDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data  # list of [question, correct_answer, reasoning_string]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question, _, reasoning = self.data[idx]
            return tokenize(self.tokenizer, question=question, answer=reasoning)

    train_data = RFTDataset(llm.tokenizer, rft_data)
    valid_data = TokenizedDataset(
        llm.tokenizer,
        Dataset("valid"),
        lambda q, a: {"question": q, "answer": f"<answer>{round(a, 3)}</answer>"}
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )

    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"RFT model saved to {output_dir}")

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})