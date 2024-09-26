def custom_combine(scores):
    # Example: Weighted average with more weight on the first score
    weights = [0.5, 0.3, 0.2]
    return sum(score * weight for score, weight in zip(scores, weights))
