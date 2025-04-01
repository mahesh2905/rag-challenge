import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

from ragas.evaluation import evaluate

responses = [
    {
        "question": "What should I do if my washing machine makes a loud noise?", #paste your actual question
        "response": "Please contact customer support for troubleshooting.", #paste your response
        "context": "Our customer support team can help with troubleshooting loud noises from your washing machine.", #paste your context
    }
]

result = evaluate(responses)
print(result)
