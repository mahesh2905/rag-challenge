import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "True"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from flask import Flask, request, jsonify
import torch
import re

from llama_classes import retriever
from pygame import mixer
from flask import Flask, render_template, request, url_for, redirect

from transformers import AutoModelForCausalLM,  pipeline, AutoTokenizer

app = Flask(__name__)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def generate(query, retrieved_context):
    # Formatting inputs
    system_prompt = "You are a Q&A assistant. Your goal is to answer questions accurately based on the instructions and context provided."
    formatted_context = "\n".join([doc.page_content for doc in retrieved_context])

    # Creating prompt
    prompt = f"{query}\n{formatted_context}\n"

    # Initializing pipeline
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    # Applying chat template
    messages = [
        {"role":"system","content":prompt},
        {"role": "user", "content": query}
    ]
    formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generating response
    outputs = pipe(formatted_prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]

    # Cleaning the response
    response = re.sub(r"<\|.*?\|>", "", response)  # Remove template tokens
    response = re.sub(r"</s>", "", response)        # Remove end-of-sequence tokens
    response = response.replace(system_prompt, "").strip()  # Remove system prompt

    return response


@app.route('/generate_response', methods=['GET', 'POST'])
def Rag_qaImage():
    if request.method == 'POST':
        query = request.form.get('query')
        retriever_context = retriever(query)
        print(retriever_context)
        result = generate(query, retriever_context)
        return jsonify({'response': result})
    else:
        return jsonify({'message': 'Please use a POST request to submit data.'}), 405

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5003)
