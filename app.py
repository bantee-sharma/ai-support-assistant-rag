from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from textblob import TextBlob

user_input = input("Ask anything: ")
sentiment_score = TextBlob(user_input).sentiment.polarity
print(sentiment_score)