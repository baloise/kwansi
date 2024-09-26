import dspy

class Assess_Interestingness(dspy.Signature):
    """Assess how interesting and engaging the tweet is.
    
    Scoring guide:
    0: Completely uninteresting, no engagement value
    5: Moderately interesting, some engagement potential
    10: Highly interesting, very likely to engage readers"""
    tweet = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")

class Assess_StyleAppropriateness(dspy.Signature):
    """Assess if the style of the tweet is appropriate for the topic and platform.
    
    Scoring guide:
    0: Completely inappropriate style
    5: Somewhat appropriate style
    10: Perfectly appropriate style for the topic and Twitter"""
    tweet = dspy.InputField()
    topic = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")
