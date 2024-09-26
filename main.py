import dspy
from dspy.teleprompt import BootstrapFewShot
import os
from dotenv import load_dotenv
import json
from src.data_preparation import prepare_examples
from datetime import datetime
from signatures.assessors import Assess_Interestingness, Assess_StyleAppropriateness
from signatures.task import TweetCreatorSignature

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
tweet_examples = prepare_examples(data, input_fields, n_samples=10)


class TweetCreator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.creator = dspy.ChainOfThought(TweetCreatorSignature)

    def forward(self, topic, details):
        return self.creator(topic=topic, details=details)

tweet_creator = TweetCreator()

def metric(example, pred, trace=None):
    tweet = pred.tweet
    topic = example.topic

    with dspy.context():
        interestingness_score = dspy.Predict(Assess_Interestingness)(tweet=tweet)
        style_score = dspy.Predict(Assess_StyleAppropriateness)(tweet=tweet, topic=topic)

    # Function to extract numerical score from string
    def extract_score(score_str):
        try:
            return float(score_str.split(':')[-1].strip())
        except ValueError:
            return 0  # Return 0 if conversion fails

    # Convert scores to floats between 0 and 1
    interestingness_score = extract_score(interestingness_score.score) / 10
    style_score = extract_score(style_score.score) / 10

    # Ensure the tweet is within the character limit
    length_score = 1 if len(tweet) <= 280 else 0

    total_score = interestingness_score * style_score * length_score

    if trace is not None:
        return total_score >= 0.25  # During compilation, accept if score is 0.5 or higher
    
    return total_score

# Initialize optimizer
optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=5
)

# Compile optimizer
optimized_tweet_creator = optimizer.compile(student=tweet_creator, trainset=tweet_examples)

# save the optimized tweet creator
optimized_tweet_creator.save('output/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_optimized_tweet_creator.json')

# Test the Optimized Program
for idx, example in enumerate(tweet_examples[:5]):  # Test with first 5 examples
    prediction = optimized_tweet_creator(topic=example.topic, details=example.details)
    print(f"Tweet {idx+1}:")
    print(f"Topic: {example.topic}")
    print(f"Details: {example.details}")
    print(f"Generated Tweet: {prediction.tweet}")
    print(f"Character Count: {len(prediction.tweet)}\n")
