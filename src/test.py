import openai
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#Function to read documents
def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents

# Passing the directory to the 'load_docs' function
directory = '../reference/test'
documents = load_docs(directory)
len(documents)

#This function will split the documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

query_result = embeddings.embed_query("Hello Buddy")
#print(len(query_result),query_result)


import os
from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_3QWaDj_TirpHGJsf63e6WzcY7ZmafYmFv7dZCDYzuw5YZwEYr4NRRbZxiBkBExFqbGJQ1p")
index = pc.Index("multiple-choise")

# Example of how to add documents and embeddings to the index
index.upsert_from_documents(documents=docs, embeddings=embeddings)


#This function will help us in fetching the top relevent documents from our vector store - Pinecone
def get_similiar_docs(query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})

print(llm)