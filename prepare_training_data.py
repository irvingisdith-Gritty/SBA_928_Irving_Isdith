import pandas as pd

# ==========================================
# SBA 928 - Prepare FLAN-T5 Training Data
# ==========================================

# Load movie dataset
file_path = "Top Movies dataset.csv"
df = pd.read_csv(file_path)

print("Original dataset size:", len(df))

# Keep only the columns needed for the experiment
columns = [
    "Title",
    "Popularity",
    "Vote_Count",
    "Vote_Average",
    "Original_Language",
    "Genre",
]

df = df[columns].dropna()

# Use a small sample because training will run on CPU
df = df.head(100)

training_examples = []

for _, row in df.iterrows():
    prompt = (
        f"Analyze the market performance of the movie {row['Title']}. "
        "Use its genre, popularity, vote count, and audience rating."
    )

    response = (
        f"{row['Title']} belongs to the {row['Genre']} genre. "
        f"It has a popularity score of {row['Popularity']}, "
        f"a vote count of {row['Vote_Count']}, "
        f"and an average audience rating of {row['Vote_Average']}. "
        f"Its original language is {row['Original_Language']}. "
        "These measurements can be used to evaluate audience interest "
        "and market performance."
    )

    training_examples.append({
        "input_text": prompt,
        "target_text": response,
    })

# Convert training examples to a DataFrame
training_df = pd.DataFrame(training_examples)

# Save the training dataset
output_path = "flan_t5_training_data.csv"
training_df.to_csv(output_path, index=False)

print("\nTraining examples created:", len(training_df))

if not training_df.empty:
    print("\nExample Training Prompt:")
    print(training_df.iloc[0]["input_text"])

    print("\nExpected Response:")
    print(training_df.iloc[0]["target_text"])
else:
    print("\nNo training examples were created; check the dataset for missing values.")

print("\nTraining data saved as:")
print(output_path)