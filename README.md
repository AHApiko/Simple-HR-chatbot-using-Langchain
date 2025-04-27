# Simple-HR-chatbot-using-Langchain
A simple chatbot that answers questions about Spanish labor laws, specificallly the "Estatuto de los Trabajadores" and the "Convenio de la Publicidad". It gets answers directly from these legal texts that are public and accesible to everyone and responds in Spanish or English using OpenAI's GPT models.

It uses LangChain for document retrieval and processing, and Streamlit for the web interface. I think it goes without saying, but:

*** Don't use this chatbot as your legal advisor, always check with your HRBP or your lawyer. As I said, I'm a begginer so this chatbot can make mistakes.***

I have years of experience in HR, and I'm passionate about the data. I decided to combine my work and my passion to build something cool and useful, something that makes my everyday working life easier and that let's me practice and learn.

## Features
Upload and search HR documents. In this case, "Estatuto de los trabajadores" (labour law), "Convenio de la publicidad" (Adevrtising collective agreement).

Instant answers powered by OpenAI (GPT-3.5-turbo).

Document chunking and retrieval with FAISS.

Interactive Streamlit user interface.

## How it works
Loads HR-related documents previously manually cleaned and transformed to .txt file.

Splits the documents into smaller chunks for better retrieval.

Embeds and stores the chunks in a FAISS vector database.

When a user asks a question, the chatbot searches the database for relevant chunks and generates an answer.

## Set up

OpenAI API Key needed (https://platform.openai.com/api-keys)

Run the app (bash)
streamlit run HR_Chatbot.py

## Documents
Clean the documents to be used by removing speial characters.

If you use mine, make sure they are in the same folder as the code.

You can replace or add the documents you want to adapt it to your needs.

## Acknowledgments

Parts of the sidebar and UI were adapted from https://github.com/streamlit/llm-examples, licensed under the Apache 2.0 License.

