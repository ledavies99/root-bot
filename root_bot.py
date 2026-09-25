import logging
import os
import warnings

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


@st.cache_resource
def get_vector_store(
    folder_path: str = "./root_rules_data", db_dir: str = "./root_vector_db"
):
    """
    Checks if ChromaDB exists locally. If not, parses Markdown files from
    folder_path and builds the persistent vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(db_dir) and os.listdir(db_dir):
        print(f"📦 Loading existing Chroma vector store from '{db_dir}'...")
        return Chroma(persist_directory=db_dir, embedding_function=embeddings)

    print(
        f"🤖 Vector database not found. Scanning '{folder_path}' to build a new one..."
    )

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()

                    doc = Document(page_content=text_content, metadata={"source": file})
                    documents.append(doc)
                except (OSError, UnicodeDecodeError) as e:
                    print(f"⚠️ Error loading {file}: {e}")

    if not documents:
        print("⚠️ Warning: No rule files found! Creating fallback initial document.")
        documents.append(
            Document(
                page_content="# Rule 1.1.2 Use of CANNOT\nThe term CANNOT is absolute.",
                metadata={"source": "core_rules.md"},
            )
        )

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", "* ", " ", ""]
    )

    header_split_docs = []
    for doc in documents:
        md_chunks = markdown_splitter.split_text(doc.page_content)
        for md_doc in md_chunks:
            md_doc.metadata["source"] = doc.metadata["source"]
            header_split_docs.append(md_doc)

    chunks = text_splitter.split_documents(header_split_docs)

    print(f"📥 Building Chroma database with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_dir
    )

    return vector_store


def initialize_directory_rules_bot(
    folder_path: str = "./root_rules_data", db_dir: str = "./root_vector_db"
):
    vector_store = get_vector_store(folder_path=folder_path, db_dir=db_dir)

    active_key = (
        st.secrets.get("GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not active_key:
        raise ValueError(
            "❌ GEMINI_API_KEY is not set in Streamlit Secrets, environment variables, or .env file!"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0, google_api_key=active_key
    )

    system_prompt = (
        "You are an expert, strict rules lawyer chatbot for the board game 'Root'.\n"
        "Your job is to answer rules questions accurately using the provided text segments.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Apply strict, literal board game logic to the context provided.\n"
        "2. Cite the source file and specific rule section provided in the [Source: ... | Section: ...] headers when justifying your answer.\n"
        "3. If multiple rules interact, explain their intersection step-by-step using the provided text.\n"
        "4. If the context completely lacks information to address the question, only then state that you cannot find an official ruling.\n"
        "5. PREREQUISITE CHAIN ANALYSIS: When evaluating if a major faction action (like an Alliance Revolt) can be performed in a restricted clearing (like the Keep), you must work backward and check the prerequisite board state first:\n"
        "   - Step A: Identify the required state of the clearing to even declare the action (e.g., Rule 8.4.1 requires the clearing to be 'sympathetic').\n"
        "   - Step B: Identify how a clearing achieves that state (e.g., placing a sympathy token via Rule 8.4.2).\n"
        "   - Step C: Check if the restriction (e.g., the Keep's absolute PLACEMENT ban under Rule 6.2.2) blocks that foundational prerequisite.\n"
        "   - If the prerequisite piece cannot be placed to set up the board state, conclude immediately that the final action is impossible, bypassing any arguments about the action's mid-resolution steps.\n\n"
        "6. ADVANCED SETUP (AdSet) REQUESTS: When asked how to set up a specific faction or full game using Advanced Setup: "
        "   - Always distinguish between the Universal Draft Phase (reverse turn order, dealing 5 cards, drafting from Pool + 1) "
        "     and the Faction-Specific AdSet steps. "
        "   - List the steps sequentially as a clear numbered list. "
        "   - Remind the player explicitly that in AdSet, discarding down from 5 cards to 3 happens IMMEDIATELY after executing their faction's board placement.\n\n"
        "Context:\n{context}\n\nInput:\n{input}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    def run_rag_pipeline(user_query: str):
        docs = vector_store.search(user_query, search_type="mmr", k=10, fetch_k=25)

        formatted_chunks = []
        for doc in docs:
            source_file = doc.metadata.get("source", "Unknown Source")

            headers = [
                doc.metadata.get(h)
                for h in ["Header 1", "Header 2", "Header 3"]
                if doc.metadata.get(h)
            ]
            header_path = " > ".join(headers) if headers else "General Section"

            chunk_str = (
                f"[Source: {source_file} | Section: {header_path}]\n{doc.page_content}"
            )
            formatted_chunks.append(chunk_str)

        context_text = "\n\n---\n\n".join(formatted_chunks)

        formatted_prompt = prompt.format_messages(
            context=context_text, input=user_query
        )
        ai_response = llm.invoke(formatted_prompt)
        return ai_response.content

    return run_rag_pipeline


if __name__ == "__main__":
    RULES_DIR = "./root_rules_data"
    bot_query_function = initialize_directory_rules_bot(RULES_DIR)

    print("\n--- Folder-Backed Root Rules Bot Active ---")
    while True:
        user_query = input("Ask a rules question (or type 'exit'): ")
        if user_query.strip().lower() == "exit":
            break
        if not user_query.strip():
            continue

        answer = bot_query_function(user_query)
        print(f"\n⚖️ ANSWER:\n{answer}\n")
