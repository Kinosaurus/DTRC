# 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user

import os
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_classic.retrievers import MultiQueryRetriever
from typing import List

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

DATA_PATH = Path("./data")
CHROMA_DB_PATH = Path("./chroma_db")

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def file_ingester(file_path: str) -> List[Document]:
    try:
        loader = PyPDFDirectoryLoader(path=file_path, glob="*.pdf")
        data = loader.load()
        print("Data ingestion completed.")
        return data
    except:
        print("The file uploaded was not in PDF format, please upload a PDF file.")
        return []

def extract_into_chunks(parsed_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        strip_whitespace=True
    )
    chunks = text_splitter.split_documents(parsed_data) 
    print("Documents have been successfully split into chunks.")
    return chunks

def embed_and_store(chunks: List[Document], embed_model: str="gemini-embedding-2") -> Chroma:
    # Instantiate embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model=embed_model)
    print("Passing chunks into embedding model...")
    if CHROMA_DB_PATH.exists():
        # Setup vector database
        vector_db = Chroma(
            embedding_function=embedding_model,
            collection_name="simple-rag",
            persist_directory=CHROMA_DB_PATH
        )
        print("Vector database has been restored from local device.")
    else:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="simple-rag",
            persist_directory=CHROMA_DB_PATH
        )
        print("Vector database has been successfully setup.")
    return vector_db

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(base_retriever: Chroma, model: str="gemini-2.5-flash") -> Runnable:

    # Setup query prompt -> form internal context for llm
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
    )

    # Setup LLM
    llm = GoogleGenerativeAI(model=model)

    # Setup base retriever
    base_retriever = base_retriever.as_retriever(
        search_type="similarity"
    )

    # Setup main retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=query_prompt
    )

    # Setup RAG prompt
    rag_template = """Answer the question: {question}, based ONLY on the following context: {context}."""
    rag_prompt = ChatPromptTemplate.from_template(template=rag_template)

    # Chaining together
    chain = (
        {"context": retriever | format_docs, # knows 1. DB to get data from, 2.  LLM to use, 3. Query prompt to customise the RAG's behaviour and actions
         "question": RunnablePassthrough()}
         | rag_prompt
         | llm
         | StrOutputParser()
    )
    return chain

def perform_rag(rag_chain: Runnable, 
                user_query: str) -> str:
    res = rag_chain.invoke(
        input=(user_query)
    )
    return res

def main():
    # Ingest PDF File
    data = file_ingester(DATA_PATH)
    # Split into chunks (chunkify)
    chunks = extract_into_chunks(data)
    # Embed and store chunks into vector database
    vector_db = embed_and_store(embed_model=EMBEDDING_MODEL,
                                chunks=chunks)
    # print("Number of documents in Chroma:", vector_db._collection.count())
    # Setup retrieval mechanism
    rag_chain = create_rag_chain(model=MODEL_NAME, 
                                 base_retriever=vector_db)
    # Perform RAG and return answer to user query
    
    # # Create streamlit title
    st.title("Driving Theory RAG")
    with st.form("my_form"):
        user_query = st.text_area(
            "Enter your query here:",
            "Your message..."
        )
        submitted = st.form_submit_button("Ask")
        if submitted:
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Generating answer..."):
                  answer = perform_rag(rag_chain, user_query)
                st.write(answer)

if __name__ == "__main__":
    main()
