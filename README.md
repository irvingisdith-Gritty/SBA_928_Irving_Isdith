# SBA 928 - Movie Market Research with FLAN-T5

## Project Overview
This project investigates fine-tuning Google's FLAN-T5-small model for movie market research using the Top Movies dataset.

## Dataset
- Source: Top Movies dataset.csv
- Complete movie records: 9,826
- Fine-tuning examples: 600
- Training examples: 480
- Validation examples: 120
- Split seed: 42

## Six Task Families
1. Consumer Behavior
2. Genre Trends
3. International Markets
4. Audience Segmentation
5. Competitor Comparison
6. Recommendation Strategy

## Project Files
- prepare_training_data.py: Generates the training dataset.
- flan_t5_training_data.csv: Fine-tuning examples.
- train_flan_t5.py: Initial training experiment.
- train_flan_t5_3epochs.py: Three-epoch training experiment.
- compare_models.py: Model comparison script.
- model_comparison_results_4models.csv: Four-model comparison outputs.

## Models
- google/flan-t5-small: Original pretrained baseline.
- flan_t5_movie_model: Initial fine-tuned model.
- flan_t5_movie_model_600: Earlier 600-example model.
- flan_t5_movie_model_600_diverse: Diversified 600-example model.
- flan_t5_movie_model_600_diverse_3epochs: Three-epoch diversified model.

## Three-Epoch Training Configuration
- Base model: google/flan-t5-small
- Epochs: 3
- Batch size: 2
- Learning rate: 0.00005
- Input maximum length: 128 tokens
- Target maximum length: 128 tokens
- Hardware: CPU
- Framework: Hugging Face Transformers

## Validation Results
| Epoch | Validation Loss |
|-------|-----------------|
| 1 | 1.5218 |
| 2 | 0.9288 |
| 3 | 0.7733 |

Validation loss decreased across all three epochs.

## Evaluation
The project includes independent prompts across six market-research task families and comparisons with the original FLAN-T5 model.

The model demonstrated some ability to reproduce movie statistics and answer simple questions. However, generated responses sometimes contained repetition, unsupported conclusions, or incomplete analytical explanations.

Lower validation loss did not consistently translate into improved generation quality.

## Limitations and Responsible AI
- Popularity and vote counts do not establish purchase behavior.
- Ratings are limited indicators of audience satisfaction.
- Original language does not establish viewer location.
- One film cannot establish a genre-wide trend.
- Movie metadata does not identify viewer demographics.
- Marketing recommendations require further validation.
- Model-generated conclusions should be reviewed for factual accuracy and unsupported assumptions.

## Reproduction
Install the dependencies required by the project, then run:

1. python prepare_training_data.py
2. python train_flan_t5_3epochs.py
3. python compare_models.py

Training is CPU-compatible but may take several minutes.

## Conclusion
Fine-tuning reduced validation loss substantially, but independent generation tests revealed continuing limitations in analytical reasoning. The results demonstrate the importance of evaluating generated content rather than relying exclusively on training metrics.
