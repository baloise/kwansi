import dspy
from typing import List, Callable, Tuple, Union

def create_metric(
    assessors: List[Tuple[Callable, dict]], 
    additional_metrics: List[Callable] = None, 
    combine_method: Union[str, Callable] = 'multiplicative',
    threshold: float = 0.25
):
    def metric(example, pred, trace=None):
        scores = []
        
        # Run assessors
        for assessor, kwargs in assessors:
            with dspy.context():
                result = dspy.Predict(assessor)(**{**kwargs, **vars(example), **vars(pred)})
            score = extract_score(result.score) / 10
            scores.append(score)
        
        # Run additional metrics
        if additional_metrics:
            for metric_func in additional_metrics:
                scores.append(metric_func(example, pred))
        
        # Combine scores
        if isinstance(combine_method, str):
            if combine_method == 'multiplicative':
                total_score = 1
                for score in scores:
                    total_score *= score
            elif combine_method == 'additive':
                total_score = sum(scores) / len(scores)
            else:
                raise ValueError("Invalid combine_method string. Choose 'multiplicative' or 'additive'.")
        elif callable(combine_method):
            total_score = combine_method(scores)
        else:
            raise ValueError("combine_method must be either 'multiplicative', 'additive', or a callable function.")
        
        if trace is not None:
            return total_score >= threshold
        
        return total_score
    
    return metric

def extract_score(score_str):
    try:
        return float(score_str.split(':')[-1].strip())
    except ValueError:
        return 0  # Return 0 if conversion fails
