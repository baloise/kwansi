def length_metric(example, pred):
    return 1 if len(pred.tweet) <= 280 else 0
