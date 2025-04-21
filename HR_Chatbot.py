import os
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.chains import RetrievalQA  


st.set_page_config(page_title="HR Chatbot App", layout="wide")
st.title("HR Chatbot")
st.markdown("Ask HR questions!")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key", type="password"
)

if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

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


def main():
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question before submitting.")
            return
        with st.spinner("Setting up model..."):
            qa_chain = load_qa_chain()
        with st.spinner("Processing..."):
            result = qa_chain({"query": query})
        
        st.subheader("Answer")
        st.write(result["result"])

        
        st.subheader("Source Documents")
        for doc in result.get("source_documents", []):
            content = doc.page_content.strip().replace("\n", " ")
            excerpt = content[:200] + ("..." if len(content) > 200 else "")
            metadata = doc.metadata
            source_info = metadata.get("source", "Unknown source")
            st.markdown(f"**Source:** {source_info}")
            st.write(excerpt)
            st.write("---")

if __name__ == "__main__":
    main()
