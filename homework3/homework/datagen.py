import json
from pathlib import Path


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel()
    dataset = Dataset("train")

    results = []

    questions = [dataset[i][0] for i in range(len(dataset))]
    correct_answers = [dataset[i][1] for i in range(len(dataset))]

    # Generate `oversample` completions per question
    prompts = [model.format_prompt(q) for q in questions]
    all_generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

    for question, correct_answer, generations in zip(questions, correct_answers, all_generations):
        # Find first correct generation
        for gen in generations:
            try:
                predicted = float(gen.split("<answer>")[1].split("</answer>")[0])
            except (IndexError, ValueError):
                continue

            if is_answer_valid(predicted, correct_answer):
                # Store: [question, correct_answer, reasoning+answer string]
                results.append([question, correct_answer, gen.strip()])
                break  # Only keep one correct generation per question

    print(f"Generated {len(results)} / {len(dataset)} correct examples ({100 * len(results) / len(dataset):.1f}%)")

    # Save to json
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved dataset to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)