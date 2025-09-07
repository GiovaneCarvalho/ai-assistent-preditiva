from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.docstore.document import Document

BASE_DIR = Path().resolve().parent
DOCS_DIR = BASE_DIR / "docs"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf_vectorstore(filepath: str, save_path: str):
    loader = PyPDFLoader(DOCS_DIR / filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    documents = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(f"vectorstores/{save_path}")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 7}
    )
    return retriever


def load_excel_vectorstore(filepath: str, save_path: str):
    df = pd.read_excel(DOCS_DIR / filepath)
    documents = []
    for idx, row in df.iterrows():
        text = " ".join([str(cell) for cell in row if pd.notna(cell)])
        documents.append(Document(page_content=text, metadata={"row": idx}))

    vectorstore_tickers = FAISS.from_documents(documents, embedding)
    vectorstore_tickers.save_local("vectorstores/vectorstore_tickets")
    retriever_tickets = vectorstore_tickers.as_retriever(
        search_type="similarity", search_kwargs={"k": 7}
    )

    return retriever_tickets


retriever_perguntas_frequentes = load_pdf_vectorstore(
    "Perguntas Frequentes.pdf", "vectorstore_perguntas_frequentes"
)
retriever_manual_tecnico = load_pdf_vectorstore(
    "Manual Tecnico de Produtos.pdf", "vectorstore_manual_tecnico_produtos"
)
retriever_politicas_procedimentos = load_pdf_vectorstore(
    "Politicas e Procedimentos.pdf", "vectorstore_politicas_procedimentos"
)
retriever_tickets = load_excel_vectorstore("Tickets.xlsx", "vectorstore_tickets")
