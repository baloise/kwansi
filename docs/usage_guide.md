# Kwansi Usage Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Key Elements](#key-elements)
   - [Tasks](#tasks)
   - [Evaluators](#evaluators)
   - [Examples](#examples)
   - [Optimizers](#optimizers)
3. [Key Functions](#key-functions)
   - [prepare_examples()](#prepare_examples)
   - [create_task()](#create_task)
   - [create_evaluator()](#create_evaluator)
   - [run_optimizer()](#run_optimizer)
   - [save_optimized_model()](#save_optimized_model)
   - [test_model()](#test_model)
4. [Typical Workflow](#typical-workflow)
   - [Preparation](#preparation)
   - [Optimization](#optimization)
   - [Further Use](#further-use)

## Introduction

Kwansi is a library that makes it easy to create and optimize prompts for language models. It is built on top of the DSPy framework, and helps you to utilize DPSy's self-optimization techniques to automatically create prompts that are tailored to specific tasks.

Note that anything Kwansi can do, DSPy can do as well (and more); Kwansi just offers a simpler function set for creating optimizing existing prompts.

Please make yourself familiar with the fundamentals of [DSPy](https://dspy-docs.vercel.app/docs/intro) before using Kwansi: Signatures, Modules, Examples, Optimizers, and Metrics.

## Key Elements

To auto-optimize a prompt, you will need a **task** (an existing prompt Signature you want to optimize), one or multiple **evaluators** (ways to measure a prompt's quality), and a **dataset** (a set of Examples that you want to use to optimize the prompt). To run the optimizer, you will need to call the optimizer function with the task, evaluators, and dataset as arguments.

### Tasks

A **task** is a prompt Signature that you want to optimize. It is a definition of the input and output of the prompt, i.e. its interface. It's recommended to define these in a separate file, e.g. `tasks.py`.

### Evaluators

An **evaluator** is a way to measure a prompt's quality. It is a function that takes the input fields from a set of Examples and returns a numerical quality score (for example, a score between 0 and 100 or a binary 0/1 value). Evaluators fall into two groups:
- **Metrics**: These are simple algorithms (e.g. the length of a generated text or a regex match).
- **Assessors**: These are more complex algorithms that use language model signatures themselves to score other prompts (for example, assessing whether a generated text is interesting or customer-oriented).

You can use a **combiner** to combine multiple evaluators into a single score. There are three types of combiners:
- **Additive**: These add the scores of the evaluators together.
- **Multiplicative**: These multiply the scores of the evaluators together.
- **custom function**: You can create a custom function to combine the scores in any way you like, e.g. with weights or other parameters.

### Examples

**Example** are a JSON representation of a data point that you want to use to optimize the prompt. These can be simple one-field inputs as the following:

```json
{
    "celebrities": [
        {"name": "Tom Hanks"},
        {"name": "Beyonce"},
        {"name": "Roger Federer"}
    ]
}
```
or more complex structures with multiple fields: 

```json
{
    "customers": [
        {
            "segment": "Millennials",
            "age": "25-35",
            "interests": ["travel", "tech", "fitness"],
        },
        {
            "segment": "Boomers",
            "age": "55-65",
            "interests": ["history", "family", "gardening"],
        }
    ]
}
```

Note that these data points just represent inputs for a prompt, so it's not necessary to create "gold standard" outputs. They need to be representative of the input space you want to optimize over, and the outputs you want to generate. 

Depending on the optimizer you use, you will need different amounts of examples. Please check DSPy's documentation [Which optimizer should I use?](https://dspy-docs.vercel.app/docs/building-blocks/optimizers#which-optimizer-should-i-use) for more information.

### Optimizers

An **optimizer** is an algorithm that automatically improves a prompt. It is a function that takes a set of Examples and a metric and returns a new, improved set of Examples. There are different optimizers available, each with their own approach to auto-improving prompts:

- BootstrapFewShot: Simple and fast, based on adding few-shot examples to the end of the prompt
- BootstrapFewShotWithRandomSearch: BootstrapFewShot with additional random search elements.
- COPRO: Gradually optimizes the signature instructions to be more effective.
- MIPROv2: Optimizes both the instructions and few-shot examples at once.

**Compiled optimizers** are the output of an optimization process. They are ready to be used for inference and can be saved to a JSON file for later use.

## Key Functions

### prepare_examples()

`data_preparation.prepare_examples()` is a function that prepares a set of examples for a given task. It is used to convert a set of input data into a set of examples that can be used to optimize a prompt. Parameters:

- `data`: A set of JSON data to prepare examples from.
- `input_fields`: A dictionary that defines the fields of the input data that should be used to create the examples, consisting of a `data_key` and a list of `fields`. The key is the name of the field, and the value is the data to use for that field. Example:

  ```python
  input_fields = {
      "data_key": "celebrities",
      "fields": ["name"]
  }
  ```

- `n_samples`: An integer that defines the number of samples to prepare. If not set, the whole dataset is used.

### create_task()

`task_creation.create_task()` is a function that converts a signature into a task that can be used to optimize a prompt. Parameters:

- `signature`: A DSPy signature to create a task from.
- `module_name`: The name of the DSPy module to use for the task (usually ´Predict´ or ´ChainOfThought´).

### create_evaluator()

`metric_creation.create_evaluator()` is a function that converts a set of assessors and additional metrics into a single evaluator that can be used to optimize a prompt. Parameters:

- `assessors`: A list of assessors to use for the evaluator.
- `additional_metrics`: A list of additional metrics to use for the evaluator.
- `combine_method`: The method to use to combine the assessor scores and additional metrics into a single score.
- `threshold`: The minimum score for the evaluator to be considered (barely) passing. This value is between 0 and 1 and depends on the combiner method (for example, an additive combiner with a threshold of 0.5 means that the evaluator will be considered passing if the combined score is greater than 0.5, while a multiplicative combiner with a threshold of 0.5 means that the evaluator will be considered passing if the product of the scores of all assessors and metrics is greater than 0.5).

Example:

```python
marketing_message_evaluator = create_evaluator(
    assessors=[
        ('Relevance', Assess_Relevance, {'message': 'message', 'segment': 'segment', 'interests': 'interests'}, (0, 10)),
        ('Tone_Appropriateness', Assess_ToneAppropriateness, {'message': 'message', 'age': 'age'}, (0, 10)),
        ('Persuasiveness', Assess_Persuasiveness, {'message': 'message'}, (0, 10))
    ],
    additional_metrics=[
        ('Keyword_Inclusion', keyword_inclusion_metric),
        ('Message_Length', message_length_metric)
    ],
    combine_method="additive",
    threshold=0.5
)
```

### run_optimizer()

`optimizer_handling.run_optimizer()` is a function that runs the optimizer and returns the optimized task and the optimizer type. It takes care of DSPy's optimizer initialization and compilation process. Parameters:

- `optimizer_type`: The type of optimizer to use. This can be "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "COPRO", or "MIPROv2".
- `metric`: The metric to use for the optimizer, usually created with `create_metric()`.
- `student`: The task definition you are optimizing, usually created with `create_task()`.
- `trainset`: The trainset to use for the optimizer, usually created with `prepare_examples()`.


### save_optimized_model()

`optimizer_handling.save_optimized_model()` is a function that saves the optimized task and the optimizer type to a JSON file. Parameters:

- `optimized_model`: The optimized task, usually the output of `run_optimizer()`.
- `optimizer_type`: The type of optimizer used, usually the output of `run_optimizer()`.
- `folder`: The folder to save the optimized model to.
- `name`: The name of the optimized model.

### test_model()

`testing.test_model()` is a function that produces a set of test results for a model's performance on a set of examples. Parameters:

- `model`: The model to test, usually the output of `run_optimizer()`.
- `test_data`: The test data to use for the test, usually created with `prepare_examples()`.
- `input_fields`: The fields of the input data to show in the test results.
- `output_field`: The field of the output data to show in the test results.
- `metric`: The metric to use for the test, usually created with `create_metric()`.
- `verbose`: Whether to print verbose output during the test.
- `truncate` (only if verbose is False): The number of characters to show in the test (non-verbose) results.

## Typical Workflow

### Preparation

1. Obtain or create a dataset of input data.
2. Create a signature for the task you want to optimize.
3. Create evaluators (either metric functions or assessor signatures).
4. Optional: Create a custom combiner function for your evaluators (or choose additive / multiplicative).

### Optimization

1. Define the DSPy model you want to use with `dspy.settings.configure()`.
2. Load data, additional wrangling (if needed), make sure it's in JSON format.
3. Define the input fields in the JSON file.
4. Prepare your examples with `prepare_examples()`.
5. Create your task with `create_task()`.
6. Create your evaluators with `create_evaluator()`.
7. Run the optimizer with `run_optimizer()`.
8. Save the optimized model with `save_optimized_model()`.
9. Test the optimized model with `test_model()`.
10. Iterate on the process by refining your evaluators and repeating the process.

### Further Use

After optimization, utilize the optimized model for your specific use case in another DSPy workflow.

## Example

You can check out a full example in the separate repository here: [tbd]