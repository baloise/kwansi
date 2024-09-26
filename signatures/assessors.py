import dspy

class Assess_Interestingness(dspy.Signature):
    """Assess how interesting and engaging the tweet is on a scale of 0 (very uninteresting) to 10 (highly interesting)."""
    tweet = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10 in the format 'Score: X'")
    explanation = dspy.OutputField()

class Assess_StyleAppropriateness(dspy.Signature):
    """Assess if the style of the tweet is appropriate for the topic and platform on a scale of 0 (very inappropriate) to 10 (perfectly appropriate)."""
    tweet = dspy.InputField()
    topic = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10 in the format 'Score: X'")
    explanation = dspy.OutputField()
