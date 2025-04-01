import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import transformers
from langchain.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from datasets import load_dataset 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma

class Langchain_RAG:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false" 
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("loading dataset, this may take time to process")
        dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
        print(dataset)
        texts = [item['output'] for item in dataset['train']]
        print("<< dataset loaded")
        print("<< chunking")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])
        self.texts = text_splitter.create_documents(texts)
        print("<< chunked")
        self.get_vec_value = Chroma.from_documents(self.texts, self.embeddings)
        print("<< vector store created")

        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4})

    def __call__(self, query):
        if query is None:
            raise ValueError("Query cannot be None")
        return self.retriever.invoke(query)
retriever = Langchain_RAG()

