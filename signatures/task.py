import dspy

class TweetCreatorSignature(dspy.Signature):
    """As a social media expert, craft engaging and informative tweets based on given topics and details. Your tweets should be concise, attention-grabbing, and within the 280-character limit. Capture the essence of the topic, use language that sparks curiosity or emotion, and add a call to action when appropriate. Adjust the tone to suit the topic, convey information clearly, and emphasize timeliness if relevant. Your tweet should inform, engage, and encourage further interaction or exploration of the topic. Don't use hashtags."""

    topic = dspy.InputField(desc="the main subject of the tweet")
    details = dspy.InputField(desc="additional information or context for the tweet")

    tweet = dspy.OutputField(desc="a tweet")
