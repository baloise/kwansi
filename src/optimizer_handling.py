from dspy.teleprompt import BootstrapFewShot
from datetime import datetime
import os

def initialize_optimizer(optimizer_type='BootstrapFewShot', **kwargs):
    optimizer_classes = {
        'BootstrapFewShot': BootstrapFewShot
    }
    
    default_settings = {
        'BootstrapFewShot': {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 16,
            'max_rounds': 1,
            'max_errors': 5
        }
    }
    
    if optimizer_type not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    optimizer_class = optimizer_classes[optimizer_type]
    settings = default_settings[optimizer_type].copy()
    settings.update(kwargs)
    
    return optimizer_class(**settings)

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
    optimizer = initialize_optimizer(optimizer_type, metric=metric, **kwargs)
    optimized_model = compile_optimizer(optimizer, student, trainset)
    return optimized_model
