from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2
from datetime import datetime
import os

# Combined dictionary for optimizer classes, their default settings, and compile arguments
OPTIMIZERS = {
    'BootstrapFewShot': {
        'class': BootstrapFewShot,
        'default_settings': {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 16,
            'max_rounds': 1,
            'max_errors': 5
        },
        'compile_args': {}
    },
    'BootstrapFewShotWithRandomSearch': {
        'class': BootstrapFewShotWithRandomSearch,
        'default_settings': {
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 16,
            'max_rounds': 1,
            'num_candidate_programs': 16,
            'num_threads': 64,
            'max_errors': 10
        },
        'compile_args': {}
    },
    'COPRO': {
        'class': COPRO,
        'default_settings': {
            'breadth': 5,
            'depth': 3
        },
        'compile_args': {
            'eval_kwargs': {
                'num_threads': 64,
                'display_progress': True,
                'display_table': 0
            }
        }
    },
    'MIPROv2': {
        'class': MIPROv2,
        'default_settings': {
            'num_candidates': 7,
            'init_temperature': 0.5,
            'verbose': False,
            'num_threads': 64,
            'max_errors': 10
        },
        'compile_args': {
            'max_bootstrapped_demos': 3,
            'max_labeled_demos': 4,
            'num_trials': 15,
            'minibatch_size': 25,
            'minibatch_full_eval_steps': 10,
            'minibatch': True,
            'requires_permission_to_run': False
        },
        'min_trainset_size': 50,  # Minimum recommended size for the training set
        'min_valset_size': 50,    # Minimum recommended size for the validation set
        'valset_ratio': 0.8       # Ratio of trainset to use as valset if not provided
    }
}

def initialize_optimizer(optimizer_type='BootstrapFewShot', **kwargs):
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    optimizer_info = OPTIMIZERS[optimizer_type]
    settings = optimizer_info['default_settings'].copy()
    settings.update(kwargs)
    
    return optimizer_info['class'](**settings)

def compile_optimizer(optimizer, student, trainset, optimizer_type):
    optimizer_info = OPTIMIZERS[optimizer_type]
    compile_args = optimizer_info['compile_args'].copy()
    
    # Check trainset size
    if len(trainset) < optimizer_info.get('min_trainset_size', 0):
        raise ValueError(f"Training set size ({len(trainset)}) is too small for {optimizer_type}. Minimum recommended size is {optimizer_info['min_trainset_size']}.")
    
    # Prepare valset if not provided
    if 'valset' not in compile_args:
        valset_size = int(len(trainset) * optimizer_info.get('valset_ratio', 0.8))
        compile_args['valset'] = trainset[:valset_size]
    
    # Check valset size
    if len(compile_args['valset']) < optimizer_info.get('min_valset_size', 0):
        raise ValueError(f"Validation set size ({len(compile_args['valset'])}) is too small for {optimizer_type}. Minimum recommended size is {optimizer_info['min_valset_size']}.")
    
    # Adjust minibatch_size if necessary
    if 'minibatch_size' in compile_args and compile_args['minibatch_size'] > len(compile_args['valset']):
        compile_args['minibatch_size'] = len(compile_args['valset'])
    
    compile_args.update(student=student, trainset=trainset)
    return optimizer.compile(**compile_args)

def save_optimized_model(optimized_model, optimizer_type, folder='output', name=None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if name is None:
        name = f'{timestamp}_optimized_model_{optimizer_type}'
    else:
        name = f'{timestamp}_{name}_{optimizer_type}'
    
    filename = f'{name}.json'
    filepath = os.path.join(folder, filename)
    
    os.makedirs(folder, exist_ok=True)
    optimized_model.save(filepath)
    print(f"Optimized model saved to {filepath}")

def run_optimizer(optimizer_type, metric, student, trainset, **kwargs):
    def wrapped_metric(*args, **kwargs):
        result = metric(*args, **kwargs)
        if isinstance(result, dict):
            return sum(result.values()) / len(result)
        return result

    optimizer = initialize_optimizer(optimizer_type, metric=wrapped_metric, **kwargs)
    optimized_model = compile_optimizer(optimizer, student, trainset, optimizer_type)
    return optimized_model, optimizer_type
