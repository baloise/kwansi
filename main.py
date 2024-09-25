import dspy
from dspy.teleprompt import COPRO
import os
from dotenv import load_dotenv
import re
import json
import random
from config import STANDARD_GREETING, STANDARD_CLOSING, STANDARD_LINK

load_dotenv()

dspy.settings.configure(lm=dspy.OpenAI(
    model='gpt-4o-mini',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=1024
))

# Load the JSON file
with open('data/clean/ExportNachrichtentexteV2_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter out entries with empty messages
valid_entries = [entry for entry in data if entry.get('message') and entry['message'].strip()]

# Select 50 random samples
random_samples = random.sample(valid_entries, 50)

# Define your data
email_templates = [
    dspy.Example(
        account=sample.get('accountName', ''),
        folder=sample.get('folderName', ''),
        template_name=sample.get('description', ''),
        template_id=sample.get('_id', ''),
        mail_subject=sample.get('messageSubject', ''),
        mail_body=sample.get('message', ''),
        mail_greeting=STANDARD_GREETING,
        mail_closing=STANDARD_CLOSING,
        mail_link=STANDARD_LINK
    ).with_inputs("account", "folder", "template_name", "template_id", "mail_subject", "mail_body", "mail_greeting", "mail_closing", "mail_link")
    for sample in random_samples
]

class Assess_ContentAccuracy(dspy.Signature):
    """Assess the accuracy of the template explanation.
    
    Scoring guide:
    0: Completely inaccurate, no relevant information provided
    2: Mostly inaccurate with only a few correct details
    4: Partially accurate, but with significant omissions or errors
    6: Mostly accurate with minor omissions or errors
    8: Highly accurate with very few minor inaccuracies
    10: Perfectly accurate, capturing all key information without errors"""
    template_explanation = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")

class Assess_AudienceIdentification(dspy.Signature):
    """Assess how well the recipient role is tailored for a copywriter's understanding of the context the email template is intended for.
    
    Scoring guide:
    0: Completely misidentifies the recipient role or provides no information
    2: Vaguely identifies a role but is mostly incorrect or unhelpful
    4: Identifies a general role but lacks specificity needed for tailoring
    6: Correctly identifies the role with some useful details for tailoring
    8: Accurately identifies the role with clear implications for tailoring
    10: Perfectly identifies the role with excellent insights for tailoring language and tone"""
    recipient_role = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")

class Assess_ContextCompleteness(dspy.Signature):
    """Assess the completeness of the context provided in the summary.
    
    Scoring guide:
    0: No context provided, completely incomplete
    2: Minimal context, missing most crucial information
    4: Some context provided, but significant gaps remain
    6: Good context with most key points covered, minor omissions
    8: Very comprehensive context with only trivial details missing
    10: Perfectly complete context, covering all relevant aspects"""
    template_explanation = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")

class Assess_IDAccuracy(dspy.Signature):
    """Assess if the analyzed template ID matches the actual template ID."""
    analyzed_template_id = dspy.InputField()
    actual_template_id = dspy.InputField()
    score = dspy.OutputField(desc="A score of either 0 = 'no match' or 1 = 'match'")

class Assess_ContextUnderstanding(dspy.Signature):
    """Assess the overall utility of the summary for a copywriter's understanding of the context of the email template.
    
    Scoring guide:
    0: Provides no useful information about the context
    2: Minimal context information, leaving many questions unanswered
    4: Some context provided, but significant gaps in understanding remain
    6: Good context information, providing a solid foundation for understanding
    8: High-quality context information, leaving only minor questions unanswered
    10: Excellent and comprehensive context information, providing a complete understanding"""
    template_explanation = dspy.InputField()
    recipient_role = dspy.InputField()
    score = dspy.OutputField(desc="A score between 0 and 10")

class Assess_OutputLanguage(dspy.Signature):
    """Assess if the output is in German."""
    template_explanation = dspy.InputField()
    recipient_role = dspy.InputField()
    score = dspy.OutputField(desc="A score of either 0 = 'not in German' or 1 = 'in German'")

class Assess_FactsOnly(dspy.Signature):
    """Assess if the output only contains facts and context information and does not include recommendations or tips on how to improve the email template."""
    template_explanation = dspy.InputField()
    recipient_role = dspy.InputField()
    score = dspy.OutputField(desc="A score of either 0 = 'contains recommendations' or 1 = 'contains only facts and context'")

def summary_quality_metric(example, pred, trace=None):
    template_explanation = pred.template_explanation
    recipient_role = pred.recipient_role
    analyzed_template_id = pred.analyzed_template_id

    with dspy.context():
        content_score = dspy.Predict(Assess_ContentAccuracy)(template_explanation=template_explanation)
        audience_score = dspy.Predict(Assess_AudienceIdentification)(recipient_role=recipient_role)
        context_score = dspy.Predict(Assess_ContextCompleteness)(template_explanation=template_explanation)
        id_score = dspy.Predict(Assess_IDAccuracy)(analyzed_template_id=analyzed_template_id, actual_template_id=example.template_id)
        context_understanding_score = dspy.Predict(Assess_ContextUnderstanding)(template_explanation=template_explanation, recipient_role=recipient_role)
        language_score = dspy.Predict(Assess_OutputLanguage)(template_explanation=template_explanation, recipient_role=recipient_role)
        facts_only_score = dspy.Predict(Assess_FactsOnly)(template_explanation=template_explanation, recipient_role=recipient_role)

    # Function to extract numerical score from string
    def extract_score(score_str):
        try:
            return float(score_str.split(':')[-1].strip())
        except ValueError:
            return 0  # Return 0 if conversion fails

    # Convert scores to floats between 0 and 1
    content_score = extract_score(content_score.score) / 10
    audience_score = extract_score(audience_score.score) / 10
    context_score = extract_score(context_score.score) / 10
    context_understanding_score = extract_score(context_understanding_score.score) / 10
    
    # ID, language, and facts_only scores are already 0 or 1
    id_score = extract_score(id_score.score)
    language_score = extract_score(language_score.score)
    facts_only_score = extract_score(facts_only_score.score)

    total_score = content_score * audience_score * context_score * context_understanding_score * id_score * language_score * facts_only_score

    if trace is not None:
        return total_score >= 0.4  # During compilation, accept if score is 0.4 or higher
    
    return total_score

class EmailTemplateSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(self.signature())

    def forward(self, account, folder, template_name, template_id, mail_subject, mail_body, mail_greeting, mail_closing, mail_link):
        return self.summarizer(account=account, folder=folder, template_name=template_name, template_id=template_id,
                               mail_subject=mail_subject, mail_body=mail_body, mail_greeting=mail_greeting,
                               mail_closing=mail_closing, mail_link=mail_link)

    def signature(self):
        class EmailTemplateSummarizerSignature(dspy.Signature):
            """As a dedicated analytical assistant, your mission is to perform a comprehensive evaluation of email templates from a Swiss insurance company's customer management system. For each template, extract and synthesize vital components that convey the email's intent, audience, and actionable steps for the customer. Explain the context of the email, but do not offer any recommendations or tips on how to improve it.Focus on the following dimensions:
            
            1. **Contextual Framework**: Identify the customer service area and folder to enrich the context of the email, considering the following terms and abbreviations:
                - BDL: Baloise Direct Line, a telephone sales department
                - GA: Generalagentur, a branch office
                - KS EL: Kundenservice Einzelleben, a customer service department for life insurance (B2C)
                - KS KL: Kundenservice Kollektivleben, a customer service department for collective life insurance (B2B)
                - KS NL: Kundenservice Nichtleben, a customer service department for non-life insurance
                - KS NL TCS: Kundenservice Nichtleben Touring Club Schweiz, a customer service department for a collaboration with a roadside assistance provider
                - KS UK: Kundenservice Unternehmenskunden, a customer service department for corporate clients
                - KSS: Kundenservice Schaden, a customer service department for claims
                - LC AH/MF: Leistungscenter Autohaftpflicht/Motorfahrzeuge, a customer service department for vehicle insurance
                - LC Einzelleben: Leistungscenter Einzelleben, a customer service department for life insurance
                - LC Kollektiv Leben: Leistungscenter Kollektivleben, a customer service department for collective life insurance
                - Superaccount: A system-wide term to describe a folder type that bundles multiple folders together.
                - Schaden: A term for the claims department.
                - ADOS: A term for the company's customer data management system.
                - Perspectiva: The name of a pension solution.
            2. **Purpose Elucidation**: Clearly distill the email's intent by analyzing the subject line and body content, emphasizing key themes and messages.
            3. **Audience Profiling**: Deduce the likely role of the recipient, considering the content's language and tone, and explicitly state the intended audience.
            4. **Communication Elements**: Highlight the standard greeting and closing, ensuring that customer reassurance and clarity are prioritized.
            5. **Actionable Guidance**: Explicitly outline any actions the customer is encouraged to take, including any interactive links provided.
            
            Your analysis should be succinct, organized, and strictly based on the information provided. If any components are ambiguous or missing, clearly note these limitations without speculation. Additionally, offer insights on the email's tone and its potential impact on customer perception, aiming to foster a deeper understanding and engagement. Make sure all output is in German."""

            account = dspy.InputField(desc="The customer service area of the email template.")
            folder = dspy.InputField(desc="The folder name of the email template within the correspondence system.")
            template_name = dspy.InputField(desc="The name of the email template.")
            template_id = dspy.InputField(desc="The ID of the email template.")
            mail_subject = dspy.InputField(desc="The subject line of the email.")
            mail_body = dspy.InputField(desc="The body content of the email.")
            mail_greeting = dspy.InputField(desc="The standard greeting of the email.")
            mail_closing = dspy.InputField(desc="The standard closing of the email.")
            mail_link = dspy.InputField(desc="The standard link that the customer should click to proceed.")

            analyzed_template_id = dspy.OutputField(desc="The template ID of the email template.")
            template_explanation = dspy.OutputField(desc="A comprehensive explanation of the email's main purpose and content, and context, tailored for a copywriter's understanding of the context the email template is intended for.")
            recipient_role = dspy.OutputField(desc="The most probable role of the recipient based on the email's content and context, tailored for a copywriter's understanding of the context the email template is intended for.")

        return EmailTemplateSummarizerSignature

summarizer = EmailTemplateSummarizer()

# Initialize COPRO
optimizer = COPRO(
    metric=summary_quality_metric,
    breadth=5,
    depth=3,
    init_temperature=0.7,
    verbose=True
)

# Compile with COPRO
eval_kwargs = dict(num_threads=4, display_progress=True, display_table=0)
optimized_summarizer = optimizer.compile(summarizer, trainset=email_templates, eval_kwargs=eval_kwargs)

# Print the optimized signature
print("Optimized Signature:")
signature = optimized_summarizer.signature()
print(signature.__doc__ if hasattr(signature, '__doc__') else "No signature docstring available")

# Safely print field information
for field_name in ['account', 'folder', 'template_name', 'template_id', 'mail_subject', 'mail_body', 'mail_greeting', 'mail_closing', 'mail_link', 'analyzed_template_id', 'template_explanation', 'recipient_role']:
    if hasattr(signature, field_name):
        field = getattr(signature, field_name)
        print(f"{field_name.capitalize()} field:", field.__doc__ if hasattr(field, '__doc__') else f"No {field_name} field docstring available")
    else:
        print(f"{field_name.capitalize()} field: Not available in the signature")

# Print the internal state of the summarizer (if available)
if hasattr(optimized_summarizer, 'summarizer'):
    if hasattr(optimized_summarizer.summarizer, '_demos'):
        print("\nOptimized Demonstrations:")
        for demo in optimized_summarizer.summarizer._demos:
            print(demo)
    else:
        print("\nNo optimized demonstrations available")

    if hasattr(optimized_summarizer.summarizer, '_prompt'):
        print("\nOptimized Prompt:")
        print(optimized_summarizer.summarizer._prompt)
    else:
        print("\nNo optimized prompt available")
else:
    print("\nNo internal summarizer attribute available")

# Save the optimized summarizer using DSPy's save method
save_path = 'optimized_summarizer.json'
optimized_summarizer.save(save_path)

print(f"\nOptimized summarizer saved to '{save_path}'")

# Test the Optimized Program
for idx, example in enumerate(email_templates[:5]):  # Test with first 5 examples
    prediction = optimized_summarizer(account=example.account, folder=example.folder, template_name=example.template_name, 
                                      template_id=example.template_id, mail_subject=example.mail_subject, 
                                      mail_body=example.mail_body, mail_greeting=example.mail_greeting,
                                      mail_closing=example.mail_closing, mail_link=example.mail_link)
    score = summary_quality_metric(example, prediction)
    print(f"Email Template {idx+1}:")
    print(f"Analyzed Template ID: {prediction.analyzed_template_id}")
    print(f"Template Explanation: {prediction.template_explanation}")
    print(f"Recipient Role: {prediction.recipient_role}")
    print(f"Score: {score:.2f}/1.00\n")

# To load the optimized summarizer later, you can use:
# loaded_summarizer = EmailTemplateSummarizer()
# loaded_summarizer.load(path='optimized_summarizer.json')
