import json
import os
import re
import pytesseract
import pdfplumber

from pdf2image import convert_from_path
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class ScannedPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        images = convert_from_path(self.file_path)
        text = ""
        for image in images:
            ocr_text= pytesseract.image_to_string(image,lang='chi_tra+eng')
            # Clean up extra spaces and line breaks
            cleaned_text = re.sub(r'\s+', ' ', ocr_text).strip()
            text+=cleaned_text + "\n"
        return text

    def load_and_split(self, splitter):
        text = self.load()
        print(len(text))
        return splitter.split_text(text)
def is_scanned_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():  # Check if any text was extracted
                return False  # The PDF is text-based
    return True  # No text was found, so it is likely a scanned PDF
import os

question_path="../dataset/preliminary/"

with open("../config.json") as f:
    os.environ["OPENAI_API_KEY"] = json.load(f)["key"] 
with open(os.path.join(question_path,"questions_example.json"), 'rb') as f:
    qs_ref = json.load(f)  # 讀取問題檔案
with open(os.path.join(question_path,"ground_truths_example.json"), 'rb') as f:
    gt_ref = json.load(f)


file_path="../reference/finance/489.pdf"

if (is_scanned_pdf(file_path)):
    loader=ScannedPDFLoader(file_path)
else:
    loader = PyPDFLoader(file_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
texts = loader.load_and_split(splitter)
print(len(texts))
# create local database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)


# # conversation chain
# qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
# chat_history = []
# while True:
#     query = input('\nQ: summary this ') 
#     if not query:
#         break
#     result = qa({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
#     print('A:', result['answer'])
#     chat_history.append((query, result['answer']))