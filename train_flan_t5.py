import pandas as pd
import torch

from datasets import Dataset
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainingArguments,
	Seq2SeqTrainer,
)

# ==========================================
# SBA 928 - Fine-Tune FLAN-T5 Small
# ==========================================

MODEL_NAME = "google/flan-t5-small"
TRAINING_FILE = "flan_t5_training_data.csv"
OUTPUT_DIR = "./flan_t5_movie_model"

print("=" * 60)
print("SBA 928 - FLAN-T5 Fine-Tuning Experiment")
print("=" * 60)
print("\nCUDA available:", torch.cuda.is_available())

# Load and validate training data.
df = pd.read_csv(TRAINING_FILE)
required_columns = {"input_text", "target_text"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
	raise ValueError(
		f"Training CSV is missing required columns: {', '.join(sorted(missing_columns))}"
	)

df = df.dropna(subset=["input_text", "target_text"]).copy()
if len(df) < 2:
	raise ValueError("At least two valid training examples are required.")
df["input_text"] = df["input_text"].astype(str)
df["target_text"] = df["target_text"].astype(str)

print("\nTraining examples loaded:", len(df))
dataset = Dataset.from_pandas(df, preserve_index=False)
dataset = dataset.train_test_split(test_size=0.20, seed=42)
print("Training examples:", len(dataset["train"]))
print("Evaluation examples:", len(dataset["test"]))

# Load FLAN-T5.
print("\nLoading FLAN-T5...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("FLAN-T5 loaded successfully.")

# Tokenize source and target text.
def preprocess_function(examples):
	model_inputs = tokenizer(
		examples["input_text"], max_length=128, truncation=True
	)
	labels = tokenizer(
		text_target=examples["target_text"], max_length=128, truncation=True
	)
	model_inputs["labels"] = labels["input_ids"]
	return model_inputs


print("\nTokenizing training data...")
tokenized_dataset = dataset.map(
	preprocess_function,
	batched=True,
	remove_columns=dataset["train"].column_names,
)
print("Tokenization complete.")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Configure and run training on CPU.
training_args = Seq2SeqTrainingArguments(
	output_dir=OUTPUT_DIR,
	num_train_epochs=1,
	per_device_train_batch_size=2,
	per_device_eval_batch_size=2,
	learning_rate=5e-5,
	eval_strategy="epoch",
	save_strategy="epoch",
	logging_steps=10,
	save_total_limit=1,
	report_to="none",
	use_cpu=True,
)

trainer = Seq2SeqTrainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_dataset["train"],
	eval_dataset=tokenized_dataset["test"],
	data_collator=data_collator,
)

print("\nStarting fine-tuning...")
print("CPU training may take some time.\n")
trainer.train()

# Evaluate and save the fine-tuned model.
print("\nEvaluating fine-tuned model...")
results = trainer.evaluate()
print("\nEvaluation Results:")
for key, value in results.items():
	print(f"{key}: {value}")

print("\nSaving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETE!")
print("=" * 60)
print("\nModel saved to:")
print(OUTPUT_DIR)
