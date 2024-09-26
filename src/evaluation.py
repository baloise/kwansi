import dspy
from typing import List, Tuple, Callable, Union
from signatures.assessors import Assess_Interestingness, Assess_StyleAppropriateness

def create_metric(
    assessors: List[Tuple[str, Callable, dict, Tuple[float, float]]], 
    additional_metrics: List[Tuple[str, Callable]] = None, 
    combine_method: Union[str, Callable] = 'multiplicative',
    threshold: float = 0.25
):
    def metric(example, pred, trace=None):
        scores = {}
        
        # Run assessors
        for name, assessor, kwargs, (scale_min, scale_max) in assessors:
            with dspy.context():
                result = dspy.Predict(assessor)(**{**kwargs, **vars(example), 'tweet': pred.tweet})
            raw_score = result.score
            score = (extract_score(raw_score) - scale_min) / (scale_max - scale_min)
            scores[name] = score
        
        # Run additional metrics
        if additional_metrics:
            for name, metric_func in additional_metrics:
                scores[name] = metric_func(example, pred)
        
        # Combine scores
        if isinstance(combine_method, str):
            if combine_method == 'multiplicative':
                total_score = 1
                for score in scores.values():
                    total_score *= score
            elif combine_method == 'additive':
                total_score = sum(scores.values()) / len(scores)
            else:
                raise ValueError("Invalid combine_method string. Choose 'multiplicative' or 'additive'.")
        elif callable(combine_method):
            total_score = combine_method(list(scores.values()))
        else:
            raise ValueError("combine_method must be either 'multiplicative', 'additive', or a callable function.")
        
        scores['Total_Score'] = total_score
        
        if trace is not None:
            return total_score >= threshold
        
        return scores
    
    return metric

def extract_score(score_str):
    try:
        return float(score_str.split(':')[-1].strip())
    except ValueError:
        return 0  # Return 0 if conversion fails
