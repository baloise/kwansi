import dspy

class TweetCreatorSignature(dspy.Signature):
    """As a social media expert, craft engaging and informative tweets based on given topics and details. Your tweets should be concise, attention-grabbing, and within the 280-character limit. Capture the essence of the topic, use language that sparks curiosity or emotion, include 1-2 relevant hashtags, and add a call to action when appropriate. Adjust the tone to suit the topic, convey information clearly, and emphasize timeliness if relevant. Your tweet should inform, engage, and encourage further interaction or exploration of the topic."""

    topic = dspy.InputField(desc="The main subject of the tweet.")
    details = dspy.InputField(desc="Additional information or context for the tweet.")

    tweet = dspy.OutputField(desc="An engaging tweet of 280 characters or less, including relevant hashtags.")
