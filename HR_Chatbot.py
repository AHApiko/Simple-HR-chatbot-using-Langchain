import os
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.chains import RetrievalQA  


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/AHApiko/Simple-HR-chatbot-using-Langchain/blob/main/HR_Chatbot.py)"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with HR-related questions?"}]


def get_file_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

@st.cache_resource
def load_qa_chain():
    paths = [
        "estatuto_trabajadores_cleaned.txt",
        "convenio_publicidad_cleaned.txt"
    ]

    docs = []
    for path in paths:
        full_path = get_file_path(path) 
        loader = TextLoader(full_path, encoding="utf-8")
        docs.extend(loader.load())

 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

  
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    qa_chain = load_qa_chain()
else:
    st.stop()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask something HR-related..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        result = qa_chain({"query": user_input})
        answer = result["result"]
        sources = result.get("source_documents", [])

    reply = answer
    if sources:
        reply += "\n\n**Sources:**\n"
        for doc in sources:
            content = doc.page_content.strip().replace("\n", " ")
            excerpt = content[:200] + ("..." if len(content) > 200 else "")
            source_info = doc.metadata.get("source", "Unknown source")
            reply += f"- {source_info}: {excerpt}\n"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
