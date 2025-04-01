# RAG-based Retrieval Implementation

## Project Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) based chatbot using TinyLlama and Langchain. The chatbot leverages document embeddings for context-aware responses and supports evaluation using RAGAS.

---

## Folder Structure
```
├── server.py                # Flask server for chatbot API
├── client.py                # Client script to interact with chatbot
├── eval.py                  # Evaluation script using RAGAS
├── llama_classes.py         # RAG setup and document retrieval logic
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Prerequisites
Ensure you have Python 3.9 or above installed. Install dependencies using:
```
pip install -r requirements.txt
```

Set the OpenAI API key for evaluation:
```
export OPENAI_API_KEY="YOUR_API_KEY"
```

---

## Running the Project
### 1. Start the Server
```
python server.py
```

### 2. Interact with the Chatbot
Run the client script to send queries:
```
python client.py
```

### 3. Evaluate Responses
Run the evaluation script:
```
python eval.py
```

---

## Expected Outputs
- Chatbot responses based on retrieved context.
- Evaluation results with RAGAS metrics.

---

## Issues and Debugging
Ensure the API keys are set and dependencies are correctly installed. Verify RAGAS and Langchain versions match your environment requirements.

