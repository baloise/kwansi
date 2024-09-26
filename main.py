import dspy
import os
from dotenv import load_dotenv
import json

from src.data_preparation import prepare_examples
from src.task_creation import create_task
from signatures.assessors import Assess_Interestingness, Assess_StyleAppropriateness
from signatures.task import TweetCreatorSignature
from src.evaluation import create_metric
from metrics.metrics import length_metric, hashtag_count_metric
from metrics.custom_combination import custom_combine
from src.optimizer_handling import run_optimizer, save_optimized_model
from src.testing import test_model

load_dotenv()

# define the language model
dspy.settings.configure(lm=dspy.LM(
    model='gpt-4o-mini',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=1024
))

# Load the JSON file
with open('data/clean/example_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the input fields in the JSON file
input_fields = {
    'data_key': 'tweet_instructions',
    'fields': ['topic', 'details']
}

# Prepare tweet examples
tweet_examples = prepare_examples(data, input_fields, n_samples=30)

# Create TweetCreator task
TweetCreator = create_task(TweetCreatorSignature, 'Predict')
tweet_creator = TweetCreator()

# Define the metric
tweet_metric = create_metric(
    assessors=[
        ('Interestingness', Assess_Interestingness, {'tweet': 'tweet'}, (0, 10)),
        ('Style_Appropriateness', Assess_StyleAppropriateness, {'tweet': 'tweet', 'topic': 'topic'}, (0, 10))
    ],
    additional_metrics=[
        ('Length_Check', length_metric),
        ('Hashtag_Count', hashtag_count_metric)
    ],
    combine_method="multiplicative",
    threshold=0.25
)

# Run the optimizer
optimized_tweet_creator = run_optimizer(
    optimizer_type='BootstrapFewShot',
    metric=tweet_metric,
    student=tweet_creator,
    trainset=tweet_examples
)

# Save the optimized model
save_optimized_model(optimized_tweet_creator, folder='output', name='tweet_creator')

# Test the Optimized Program (Short version)
print("Short Test Output:")
test_model(
    model=optimized_tweet_creator,
    test_data=tweet_examples,
    n_tests=5,
    input_fields=['topic', 'details'],
    output_field='tweet',
    metric=tweet_metric,
    verbose=False,
    truncate=100
)


# # Test the Optimized Program (Verbose version)
# print("Verbose Test Output:")
# test_model(
#     model=optimized_tweet_creator,
#     test_data=tweet_examples,
#     n_tests=5,
#     input_fields=['topic', 'details'],
#     output_field='tweet',
#     metric=tweet_metric,
#     verbose=True
# )
