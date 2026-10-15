
import pandas as pd

# SBA 928 - Expanded Market Research Training Dataset
# Six task families, 100 examples each

DATA_FILE = "Top Movies dataset.csv"
OUTPUT_FILE = "flan_t5_training_data.csv"

df = pd.read_csv(DATA_FILE)

columns = [
    "Title",
    "Popularity",
    "Vote_Count",
    "Vote_Average",
    "Original_Language",
    "Genre",
]

df = df[columns].dropna().copy()

# Convert numerical columns safely
for column in ["Popularity", "Vote_Count", "Vote_Average"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna().reset_index(drop=True)

print("Available movie records:", len(df))

TASK_FAMILIES = [
    "Consumer Behavior",
    "Genre Trends",
    "International Markets",
    "Audience Segmentation",
    "Competitor Comparison",
    "Recommendation Strategy",
]

# Use 100 records per task family
SAMPLES_PER_TASK = 100
TOTAL_EXAMPLES = len(TASK_FAMILIES) * SAMPLES_PER_TASK

if len(df) < TOTAL_EXAMPLES:
    raise ValueError(
        f"Need at least {TOTAL_EXAMPLES} complete movie records. "
        f"Found {len(df)}."
    )

# Random sampling for more varied movie records
sample = df.sample(
    n=TOTAL_EXAMPLES,
    random_state=42
).reset_index(drop=True)

training_examples = []
PROMPT_VARIATIONS = {
    "Consumer Behavior": [
        "Analyze observable audience-interest signals for this movie. Distinguish engagement from audience satisfaction.",
        "What do this movie's popularity, vote count, and rating suggest about audience engagement and satisfaction?",
        "Explain the audience-interest indicators for this film and identify what they cannot tell us.",
        "Assess viewer engagement and satisfaction signals without assuming individual consumer behavior.",
    ],
    "Genre Trends": [
        "Assess this movie's genre-market position using the available metrics. Explain what additional data would establish a trend.",
        "What can this film's genre and popularity metrics tell us about its market position?",
        "Explain why this movie alone cannot establish a broader genre trend.",
        "Evaluate the available genre-related evidence and identify what additional comparisons are needed.",
    ],
    "International Markets": [
        "Evaluate the movie's potential relevance to international market research. Explain the limitations of language data.",
        "What does the original language reveal about this film's international market potential?",
        "Assess the usefulness and limitations of this movie's language and popularity information.",
        "Explain what additional regional data would be necessary for an international market analysis.",
    ],
    "Audience Segmentation": [
        "Propose a cautious content-interest segment for this movie. Do not invent viewer demographics.",
        "How could this film be grouped by content interests without assuming audience demographics?",
        "Suggest a reasonable genre-based audience-interest category using the available information.",
        "Identify a potential content segment and explain the limitations of the available viewer data.",
    ],
    "Competitor Comparison": [
        "Explain how this movie could be compared with competing titles using consistent metrics. Identify missing comparison data.",
        "Which measurements would help compare this film against competing movies?",
        "Describe a fair competitor-comparison approach using this movie's available statistics.",
        "Explain why this movie's metrics alone cannot establish its competitive ranking.",
    ],
    "Recommendation Strategy": [
        "Recommend a cautious market-research strategy for this movie based on the available information.",
        "Suggest a promotional research approach based on this movie's genre and audience-interest signals.",
        "What marketing experiment could help evaluate demand for this film?",
        "Propose a testable market-research recommendation without claiming guaranteed success.",
    ],
}

for i, row in sample.iterrows():
    title = str(row["Title"])
    genre = str(row["Genre"])
    language = str(row["Original_Language"])
    popularity = row["Popularity"]
    votes = row["Vote_Count"]
    rating = row["Vote_Average"]

    task = TASK_FAMILIES[i // SAMPLES_PER_TASK]
    instruction = PROMPT_VARIATIONS[task][i % 4]
    

    context = (
        f"Movie: {title}. Genre: {genre}. "
        f"Original language: {language}. "
        f"Popularity: {popularity}. "
        f"Vote count: {votes}. "
        f"Average rating: {rating}."
    )

    if task == "Consumer Behavior":
        prompt = instruction + " " + context
    
        response = (
            f"{title} has a popularity score of {popularity}, "
            f"{votes} votes, and an average rating of {rating}. "
            "Popularity and vote count are indicators of "
            "attention and engagement, while the rating "
            "provides a limited satisfaction signal. "
            "These figures do not reveal purchase behavior "
            "or the motivations of individual viewers."
        )
        if i % 4 == 1:
           response = (
                f"For {title}, {votes} votes indicate recorded engagement, "
                f"while its average rating of {rating} provides a satisfaction signal. "
                f"The popularity score of {popularity} adds another measure of attention. "
                "These statistics cannot establish why viewers chose the movie."
            )
        elif i % 4 == 2:
            response = (
                f"{title} has {votes} recorded votes, a popularity score "
                f"of {popularity}, and an average rating of {rating}. "
                "Together, these measures offer limited evidence of "
                "audience engagement and reception. They cannot identify "
                "individual viewing habits or explain consumer motivations."
            )
        elif i % 4 == 3:
            response = (
                f"Audience-interest indicators for {title} include "
                f"a popularity score of {popularity}, {votes} votes, "
                f"and an average rating of {rating}. "
                "These measures can support preliminary market research, "
                "but they do not establish purchasing decisions, "
                "audience demographics, or individual preferences."
            )

    elif task == "Genre Trends":
        prompt = instruction + " " + context
        response = (
            f"{title} belongs to {genre}. Its popularity "
            f"is {popularity}, with {votes} votes and an "
            f"average rating of {rating}. These metrics "
            "describe one title, not a genre-wide trend. "
            "A reliable trend analysis would compare "
            "multiple films across release periods."
        )
        if i % 4 == 1:
            response = (
                f"{title} is categorized under {genre}, with a popularity "
                f"score of {popularity}, {votes} votes, and an average "
                f"rating of {rating}. These figures provide a snapshot "
                "of this film's performance. Identifying broader genre "
                "trends requires comparing multiple movies over time."
            )
        elif i % 4 == 2:
            response = (
                f"For {title}, the {genre} classification provides "
                "a starting point for genre-based market research. "
                f"With {votes} votes, a popularity score of {popularity}, "
                f"and an average rating of {rating}, the movie offers "
                "individual performance indicators. However, these "
                "numbers alone cannot establish whether its genres "
                "are growing or declining in popularity."
            )
        elif i % 4 == 3:
            response = (
                f"The film {title} belongs to the {genre} genre category. "
                f"Its popularity score of {popularity}, {votes} votes, "
                f"and average rating of {rating} offer a snapshot "
                "of audience engagement and reception. "
                "To identify meaningful genre trends, researchers "
                "should compare similar films across multiple "
                "release years and audience groups."
            )
    elif task == "International Markets":
        prompt = instruction + " " + context
        response = (
            f"{title} has original language {language}. "
            f"Its popularity is {popularity} and its "
            f"average rating is {rating}. Original "
            "language may help organize a multilingual "
            "catalog, but it does not establish where "
            "the film was watched or its international "
            "revenue. Regional viewing and sales data "
            "would be needed for market comparisons."
        )


        if i % 4 == 1:
            response = (
                f"{title} is recorded with original language {language}. "
                f"Its popularity score of {popularity}, {votes} votes, "
                f"and average rating of {rating} describe its overall "
                "performance in this dataset. These figures do not "
                "identify the countries where audiences watched it. "
                "Regional viewing data would be required."
            )
        elif i % 4 == 2:
            response = (
                f"For international market research, {title}'s original "
                f"language ({language}) is useful for catalog grouping. "
                f"The film has {votes} votes and a rating of {rating}. "
                "However, language alone cannot establish international "
                "demand, geographic reach, or overseas revenue."
            )
        elif i % 4 == 3:
            response = (
                f"The original language of {title} is {language}, "
                f"with popularity {popularity} and {votes} recorded votes. "
                "These indicators provide limited market context, "
                "not a country-by-country performance comparison. "
                "Reliable international analysis requires regional "
                "audience, distribution, or revenue figures."
            )

    elif task == "Audience Segmentation":
        prompt = instruction + " " + context
        response = (
            f"{title} could be grouped into a "
            f"{genre} content-interest segment. "
            f"Its popularity score is {popularity} "
            f"and its average rating is {rating}. "
            "This is a content-based grouping, not "
            "evidence about viewer age, income, "
            "location, or personal preferences."
        )


        if i % 4 == 1:
            response = (
                f"{title} fits a potential {genre} content-interest "
                f"segment. Its {votes} recorded votes and average "
                f"rating of {rating} provide limited engagement "
                "and reception indicators. They do not identify "
                "the demographic characteristics of its viewers."
            )
        elif i % 4 == 2:
            response = (
                f"For audience segmentation, {title} can be "
                f"categorized by its {genre} genres. "
                f"The popularity score of {popularity} and "
                f"average rating of {rating} offer preliminary "
                "performance context. Actual audience segments "
                "would require demographic or behavioral data."
            )
        elif i % 4 == 3:
            response = (
                f"The film {title} belongs to a possible "
                f"{genre} preference category, with popularity "
                f"{popularity} and {votes} recorded votes. "
                "This grouping reflects movie characteristics, "
                "not verified audience identities or preferences. "
                "Survey or viewing-history data would help "
                "validate meaningful customer segments."
            )

    elif task == "Competitor Comparison":
        prompt = instruction + " " + context
        response = (
            f"{title} has popularity {popularity}, "
            f"{votes} votes, and rating {rating}. "
            "A competitor comparison should evaluate "
            "these measurements against other films "
            "using the same data definitions and "
            "observation period. One movie record "
            "alone cannot establish competitive rank."
        )


        if i % 4 == 1:
            response = (
                f"{title} records a popularity score of {popularity}, "
                f"{votes} votes, and an average rating of {rating}. "
                "These metrics provide a baseline for comparing "
                "similar films. A meaningful competitor analysis "
                "requires matching titles by genre, release period, "
                "and consistent performance measurements."
            )
        elif i % 4 == 2:
            response = (
                f"For competitive benchmarking, {title} has "
                f"{votes} recorded votes, popularity {popularity}, "
                f"and an average rating of {rating}. "
                "These values should be compared against "
                "other relevant movies using equivalent metrics. "
                "Without competitor records, market position "
                "and relative performance remain unverified."
            )
        elif i % 4 == 3:
            response = (
                f"The competitive profile of {title} includes "
                f"a popularity score of {popularity}, {votes} votes, "
                f"and a rating of {rating}. "
                "These figures describe one film's performance, "
                "not its ranking against competitors. "
                "A reliable comparison would examine similar "
                "titles over the same observation period."
            )

    else:
        prompt = instruction + " " + context
        response = (
            f"For {title}, consider testing "
            f"{genre}-focused promotional messaging. "
            f"The popularity score of {popularity} "
            f"and rating of {rating} provide initial "
            "signals for prioritizing research. "
            "Validate the strategy using audience "
            "feedback or campaign experiments before "
            "claiming it will improve performance."
        )


        if i % 4 == 1:
            response = (
                f"For {title}, a possible marketing approach is to "
                f"test messaging centered on {genre}. "
                f"Its popularity score of {popularity} and "
                f"{votes} recorded votes offer initial engagement "
                "signals. A controlled campaign experiment would "
                "be needed to measure promotional effectiveness."
            )
        elif i % 4 == 2:
            response = (
                f"A preliminary recommendation for {title} is "
                f"to explore {genre}-focused audience campaigns. "
                f"The average rating of {rating} and popularity "
                f"score of {popularity} provide limited research "
                "context. Validate any proposed strategy through "
                "audience testing rather than assuming success."
            )
        elif i % 4 == 3:
            response = (
                f"Consider developing a promotional test for "
                f"{title} that emphasizes its {genre} characteristics. "
                f"With {votes} votes and an average rating of {rating}, "
                "the available data can help establish a starting "
                "point for market research. Campaign performance "
                "should be evaluated using measurable outcomes "
                "before recommending broader implementation."
            )

    training_examples.append({
        "task_family": task,
        "input_text": prompt,
        "target_text": response,
    })

training_df = pd.DataFrame(training_examples)

training_df.to_csv(OUTPUT_FILE, index=False)

print("\nTotal training examples:", len(training_df))
print("\nExamples by task family:")
print(training_df["task_family"].value_counts().to_string())

print("\nFirst example prompt:")
print(training_df.iloc[0]["input_text"])

print("\nFirst expected response:")
print(training_df.iloc[0]["target_text"])

print("\nSaved to:", OUTPUT_FILE)

