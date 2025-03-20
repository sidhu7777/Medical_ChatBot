from src.helper import  load_pdf_file,test_split,download_hugging_face_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os



load_dotenv


PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# Ensure loading from the correct directory
env_path = os.path.abspath(".env")  # Ensure absolute path
print(f"Loading .env from: {env_path}")

# Load .env
load_dotenv(dotenv_path=env_path)

# Get API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


extracted_data=load_pdf_file(data='Data/')
text_chunks=test_split(extracted_data)
embeddings= download_hugging_face_embedding()










pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "quickstart"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)




#Embed each chuck and upsert the embeddings into your pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)