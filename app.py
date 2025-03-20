from flask import Flask ,render_template,jsonify,request
from src.helper import download_hugging_face_embedding

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *



app = Flask(__name__)


# Ensure loading from the correct directory
env_path = os.path.abspath(".env")  # Ensure absolute path
print(f"Loading .env from: {env_path}")

# Load .env
load_dotenv(dotenv_path=env_path)

# Get API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


embeddings= download_hugging_face_embedding()


index_name = "quickstart"


#load existing index 
from langchain_pinecone import PineconeVectorStore
docsearch= PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type= 'similarity',search_kwargs={"k":3})


from langchain_openai import OpenAI
llm=OpenAI(temperature=0.4,max_tokens=500)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human", "{input}"),
    ]

)


question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])

def chat():
    try:
        msg = request.form["msg"]  # Ensure this matches frontend
        print(f"User Input: {msg}")  # Debugging

        response = rag_chain.invoke({"input": msg})
        print(f"Response: {response['answer']}")  # Debugging

        return str(response["answer"])  # Ensure returning text response
    except Exception as e:
        print(f"Error: {e}")
        return "Error processing request"



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
