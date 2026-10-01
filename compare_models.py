import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# SBA 928 - Requirement 4
# Compare Original vs Fine-Tuned FLAN-T5
# ==========================================

ORIGINAL_MODEL = "google/flan-t5-small"
FINE_TUNED_MODEL = "./flan_t5_movie_model"

print("=" * 65)
print("SBA 928 - ORIGINAL VS FINE-TUNED MODEL")
print("=" * 65)

# ------------------------------------------
# Load original model
# ------------------------------------------

print("\nLoading original FLAN-T5...")

original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
original_model = AutoModelForSeq2SeqLM.from_pretrained(ORIGINAL_MODEL)

print("Original model loaded.")

# ------------------------------------------
# Load fine-tuned model
# ------------------------------------------

print("\nLoading fine-tuned FLAN-T5...")

fine_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)
fine_model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL)

print("Fine-tuned model loaded.")

# ------------------------------------------
# Function to generate responses
# ------------------------------------------

def generate_response(model, tokenizer, prompt):

	inputs = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		max_length=128
	)

	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=100,
			num_beams=4,
			early_stopping=True
		)

	return tokenizer.decode(
		outputs[0],
		skip_special_tokens=True
	)


# ------------------------------------------
# Market research test prompts
# ------------------------------------------

test_prompts = [

	(
		"Analyze the market performance of a science fiction movie "
		"with a popularity score of 900, 5000 audience votes, "
		"and an average audience rating of 8.1."
	),

	(
		"Act as a market research analyst. Explain how movie genre, "
		"popularity, vote count, and audience rating can help a "
		"streaming company understand consumer interest."
	),

	(
		"A streaming company wants to promote highly rated action movies. "
		"Explain which movie data should be considered when making "
		"this business decision."
	)

]

# ------------------------------------------
# Compare models
# ------------------------------------------

for number, prompt in enumerate(test_prompts, start=1):

	print("\n" + "=" * 65)
	print(f"TEST PROMPT {number}")
	print("=" * 65)

	print("\nPROMPT:")
	print(prompt)

	original_response = generate_response(
		original_model,
		original_tokenizer,
		prompt
	)

	fine_response = generate_response(
		fine_model,
		fine_tokenizer,
		prompt
	)

	print("\nORIGINAL MODEL RESPONSE:")
	print(original_response)

	print("\nFINE-TUNED MODEL RESPONSE:")
	print(fine_response)


print("\n" + "=" * 65)
print("MODEL COMPARISON COMPLETE")
print("=" * 65)
