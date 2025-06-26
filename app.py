from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

user_input = input("Ask anything: ")

loader = PyPDFLoader("support_faq.pdf")
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50)
chunks = text_splitter.split_documents(doc)

embedd = HuggingFaceEmbeddings()
db = FAISS.from_documents(chunks,embedd)

retriever = db.as_retriever(search_type="mmr",kwargs={"k":3})
retriev_docs = retriever.invoke(user_input)
context = " ".join([i.page_content for i in retriev_docs])

prompt = PromptTemplate(
    template = """
You're a support agent. A customer reported the following issue:

"{ticket}"

Using the support documents below, generate a polite and helpful response:
{context}.
If required you can say reply on this mail 'xyz@support.com' and helpline number '1010100101'
""",
input_variables=["ticket","context"]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
chain = prompt | llm

response = chain.invoke({"ticket": user_input, "context": context})
print("\nGenerated Response:\n", response.content)