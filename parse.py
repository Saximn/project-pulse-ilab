#!/usr/bin/env python3
import argparse
import fitz  # PyMuPDF for PDF parsing
from tqdm import tqdm
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Ensure GPU usage: device=0 for first CUDA GPU, -1 for CPU fallback
device_index = 0 if torch.cuda.is_available() else -1

def get_pipeline_device():
    return device_index


def initialize_chain(pdf_path: str):
    """
    Build the RetrievalQA chain with summarization memory from a PDF guidebook.
    """
    # 1) Extract text from PDF
    doc = fitz.open(pdf_path)
    pages = []
    for page in tqdm(doc, desc="Extracting PDF pages"):
        pages.append(page.get_text())
    full_text = "\n\n".join(pages)

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([full_text], metadatas=[{"source": "guidebook"}])

    # 3) Embed chunks and index in FAISS
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embed_model)

    # 4) Prepare LLM pipeline (Mistral 7B)
    generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",
        device=get_pipeline_device(),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    hf_llm = HuggingFacePipeline(
        pipeline=generator,
        model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
    )

    # 5) Set up conversation memory (summary of past turns)
    memory = ConversationSummaryMemory(
        llm=hf_llm,
        memory_key="chat_history",
        summary_key="chat_summary",
        max_token_limit=800
    )

    # 6) Build the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=hf_llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False
    )
    return qa_chain


def main():
    parser = argparse.ArgumentParser(description="AI Project Management Assistant CLI")
    parser.add_argument("pdf_path", help="Path to your guidebook PDF file")
    args = parser.parse_args()

    print("Initializing knowledge base from PDF... this may take a moment.")
    qa = initialize_chain(args.pdf_path)
    print("Assistant is ready. Type 'exit' or 'quit' to stop.")

    while True:
        query = input("You: ")
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        # Query the RAG chain
        result = qa({"question": query})
        answer = result.get("answer", "No answer.")
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
