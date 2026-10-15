
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# SBA 928 - Three-Model Comparison
# 6 task families x 3 unseen prompts = 18 test prompts

MODELS = {
    "Original FLAN-T5": "google/flan-t5-small",
    "Fine-tuned 100": "./flan_t5_movie_model",
    "Fine-tuned 600": "./flan_t5_movie_model_600",
    "Fine-tuned 600 Diverse": "./flan_t5_movie_model_600_diverse",
}

OUTPUT_FILE = "model_comparison_results_4models.csv"

# These prompts are independently written test scenarios.
# They are not copied from the training CSV.
TEST_CASES = [
    ("Consumer Behavior",
     "A documentary has popularity 42, 1500 votes, and rating 8.2. "
     "Explain what these figures reveal and what they cannot reveal."),
    ("Consumer Behavior",
     "A comedy has popularity 120, 400 votes, and rating 5.9. "
     "Distinguish audience attention from satisfaction."),
    ("Consumer Behavior",
     "A drama has 9000 votes and rating 7.8. "
     "Can these numbers establish why viewers watched it?"),

    ("Genre Trends",
     "Two horror films have popularity scores 80 and 35. "
     "Can this establish that horror is growing as a genre?"),
    ("Genre Trends",
     "A science-fiction film has rating 8.4 and 3000 votes. "
     "Explain what additional evidence is needed for genre trends."),
    ("Genre Trends",
     "A streaming platform sees rising interest in animated films. "
     "Describe a defensible method for evaluating this trend."),

    ("International Markets",
     "A Spanish-language movie has popularity 95. "
     "Does its original language prove where its audience lives?"),
    ("International Markets",
     "Compare the limitations of original-language data "
     "when researching global movie demand."),
    ("International Markets",
     "A French-language film receives 2000 votes. "
     "What additional data is needed to compare regional markets?"),

    ("Audience Segmentation",
     "An action movie has rating 8.0 and popularity 110. "
     "Propose a content-interest segment without inventing demographics."),
    ("Audience Segmentation",
     "A romance film has 5000 votes. "
     "Can its audience's age or income be inferred? Explain."),
    ("Audience Segmentation",
     "Explain how genre can support content-based segmentation "
     "without establishing individual viewer preferences."),

    ("Competitor Comparison",
     "Movie A has rating 8.1 from 100 votes. "
     "Movie B has rating 7.8 from 10000 votes. "
     "Compare cautiously."),
    ("Competitor Comparison",
     "Two competing movies have popularity 80 and 120. "
     "What must be checked before declaring a market leader?"),
    ("Competitor Comparison",
     "Design a fair comparison of three streaming titles "
     "using ratings, vote counts, and popularity."),

    ("Recommendation Strategy",
     "Recommend a cautious promotion experiment for a comedy "
     "with rating 8.0 and popularity 65."),
    ("Recommendation Strategy",
     "A streaming company wants to promote highly rated thrillers. "
     "Suggest a testable recommendation strategy."),
    ("Recommendation Strategy",
     "A movie has high popularity but low ratings. "
     "Recommend a responsible marketing research approach."),
]


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


def main():
    print("=" * 65)
    print("SBA 928 - FOUR-MODEL COMPARISON")
    print("=" * 65)

    results = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()

        for index, (task_family, prompt) in enumerate(
            TEST_CASES, start=1
        ):
            response = generate_response(
                model, tokenizer, prompt
            )

            results.append({
                "model": model_name,
                "task_family": task_family,
                "test_id": index,
                "prompt": prompt,
                "response": response,
            })

            print(
                f"[{index}/{len(TEST_CASES)}] "
                f"{task_family} - completed"
            )

        del model

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "task_family",
                "test_id",
                "prompt",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 65)
    print("Models compared:", len(MODELS))
    print("Prompts per model:", len(TEST_CASES))
    print("Total responses:", len(results))
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
