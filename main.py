import dspy
import os
from dotenv import load_dotenv
import json

from src.data_preparation import prepare_examples
from src.evaluation import create_metric
from src.optimizer_handling import run_optimizer, save_optimized_model
from src.task_creation import create_task
from src.testing import test_model

########################################################
# Kwansi is a wrapper for DSPy that makes its optimizer easier to use.
# Please read the DSPy documentation to understand the key concepts:
# https://dspy-docs.vercel.app/docs/intro
# 
# This is an example implementation of how to use Kwansi to optimize a task
# You will need to:
#
# 1. set up an .env file with your OpenAI API key
# 2. have some example data in a JSON file ready
# 3. create a DSPy signature for the task you want to optimize
# 4. create one or multiple DSPy assessor signatures for the task you want to optimize: 
#    These can be prompt-based evaluators or traditional binary metrics
# 5. define a metric for the task you want to optimize (multiplicative, additive or a custom combiner)
# 6. run python main.py to start the optimization process
# 7. iteratively improve your metrics and assessors to get the best results
########################################################

# load the environment variables (especially the LLM API key)
load_dotenv()

# load custom imports (assessors, task, metrics, custom combiner)
from signatures.assessors import Assess_Interestingness, Assess_StyleAppropriateness
from signatures.task import TweetCreatorSignature
from metrics.metrics import length_metric, hashtag_count_metric
from metrics.custom_combination import custom_combine

# define the language model DSPy will use
dspy.settings.configure(lm=dspy.LM(
    model='gpt-4o-mini',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=1024
))

# Load the data - make sure to transform to JSON format
with open('data/clean/example_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the input fields in the JSON file
input_fields = {
    'data_key': 'tweet_instructions',  # the key in the JSON file that contains the data
    'fields': ['topic', 'details']  # the fields in the JSON file that are the input to the task
}

# Prepare examples (i.e. the training data)
tweet_examples = prepare_examples(data, input_fields, n_samples=30)

# Create the task (i.e. the main prompt we're trying to optimize - defined in signatures/task.py)
TweetCreator = create_task(TweetCreatorSignature, 'ChainOfThought')
tweet_creator = TweetCreator()

# Define the metric (i.e. the various evaluators of the task)
tweet_metric = create_metric(
    # Assessors are the prompt-based evaluators of the task (defined in signatures/assessors.py)
    assessors=[
        ('Interestingness', Assess_Interestingness, {'tweet': 'tweet'}, (0, 10)),
        ('Style_Appropriateness', Assess_StyleAppropriateness, {'tweet': 'tweet', 'topic': 'topic'}, (0, 10))
    ],
    # Additional metrics are traditional binary metrics (defined in metrics/metrics.py)
    additional_metrics=[
        ('Length_Check', length_metric),
        ('Hashtag_Count', hashtag_count_metric)
    ],
    # The combine method is the way we combine the assessor scores and additional metrics into a single score
    # "additive" means we add the scores, "multiplicative" means we multiply the scores
    # you can also import a custom combiner like the one in metrics/custom_combiner.py
    combine_method="multiplicative",
    # The threshold is the minimum score for the metric to be considered (barely) passing
    threshold=0.25
)

# Run the optimizer (i.e. the process of optimizing the task)
optimized_tweet_creator = run_optimizer(
    optimizer_type='BootstrapFewShot',
    metric=tweet_metric,
    student=tweet_creator,
    trainset=tweet_examples
)

# Save the optimized model - important, as you will want to reuse this optimized model in another DSPy chain
save_optimized_model(optimized_tweet_creator, folder='output', name='tweet_creator')

# Test the Optimized Program (use verbose=True for more details)
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