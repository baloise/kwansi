import random
import dspy

def prepare_examples(data, input_fields, n_samples=None):
    # Extract the relevant data based on input_fields
    examples = data.get(input_fields['data_key'], [])

    # If n_samples is not set or greater than available examples, use all
    if n_samples is None or n_samples > len(examples):
        samples = examples
    else:
        samples = random.sample(examples, n_samples)

    # Define your data
    prepared_examples = [
        dspy.Example(**{field: sample[field] for field in input_fields['fields']})
        .with_inputs(*input_fields['fields'])
        for sample in samples
    ]

    return prepared_examples