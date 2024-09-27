from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from datetime import datetime
import os

# Combined dictionary for optimizer classes and their default settings
OPTIMIZERS = {
    'BootstrapFewShot': {
        'class': BootstrapFewShot,
        'default_settings': {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 16,
            'max_rounds': 1,
            'max_errors': 5
        }
    },
    'BootstrapFewShotWithRandomSearch': {
        'class': BootstrapFewShotWithRandomSearch,
        'default_settings': {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 16,
            'max_rounds': 1,
            'num_candidate_programs': 16,
            'num_threads': 6,
            'max_errors': 10
        }
    }
}

def initialize_optimizer(optimizer_type='BootstrapFewShot', **kwargs):
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    optimizer_info = OPTIMIZERS[optimizer_type]
    settings = optimizer_info['default_settings'].copy()
    settings.update(kwargs)
    
    return optimizer_info['class'](**settings)

def compile_optimizer(optimizer, student, trainset):
    return optimizer.compile(student=student, trainset=trainset)

def save_optimized_model(optimized_model, folder='output', name=None):
    if name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        name = f'optimized_model_{timestamp}'
    
    filename = f'{name}.json'
    filepath = os.path.join(folder, filename)
    
    os.makedirs(folder, exist_ok=True)
    optimized_model.save(filepath)
    print(f"Optimized model saved to {filepath}")

def run_optimizer(optimizer_type, metric, student, trainset, **kwargs):
    # Wrap the metric in a function that ensures it returns a single numeric value
    def wrapped_metric(*args, **kwargs):
        result = metric(*args, **kwargs)
        if isinstance(result, dict):
            # If the metric returns a dictionary, we need to combine the scores
            # This is a simple example; you might need to adjust this based on your specific metric structure
            return sum(result.values()) / len(result)
        return result

    optimizer = initialize_optimizer(optimizer_type, metric=wrapped_metric, **kwargs)
    optimized_model = compile_optimizer(optimizer, student, trainset)
    return optimized_model
