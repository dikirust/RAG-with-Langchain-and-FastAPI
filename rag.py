from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
from PIL import Image
import io
import base64
import requests
import csv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (using ChatOpenAI)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")

def ocr_image_with_openai(image: Image.Image, api_key: str) -> str:
    """
    Perform OCR on a PIL image using OpenAI GPT-4o mini vision model.
    Returns the extracted text from the image.
    """
    # Convert PIL image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",  # Use GPT-4o mini vision model
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Function to set up the RAG system
def setup_rag_system():
    # Convert PDF pages to images
    pdf_path = 'data/physical monthly production 2024.pdf'
    poppler_path = r'C:\Users\diki.rustian\AppData\Local\Release-24.08.0-0\poppler-24.08.0\Library\bin'
    faiss_path = 'data/faiss_index'

    # Try to load existing FAISS vector store
    if os.path.exists(faiss_path):
        print('Loading FAISS vector store from disk...')
        vector_store = FAISS.load_local(
            faiss_path,
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True
        )
    else:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        if not images:
            raise ValueError("No images extracted from PDF. Check the file.")
        print(f"Extracted {len(images)} images from PDF.")

        # OCR each image using OpenAI Vision (gpt-4o) model
        ocr_texts = []
        for idx, img in enumerate(images):
            print(f"Performing OCR on page {idx+1} using GPT-4o Vision...")
            text = ocr_image_with_openai(img, openai_api_key)
            ocr_texts.append(text)

        # Create documents from OCR text
        documents = [Document(page_content=txt, metadata={"page": idx+1}) for idx, txt in enumerate(ocr_texts)]

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        document_chunks = splitter.split_documents(documents)
        print(f"Total document chunks created: {len(document_chunks)}")

        # Save chunks to CSV for inspection
        csv_path = 'data/ocr_chunks.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['chunk_index', 'page', 'text'])
            for i, chunk in enumerate(document_chunks):
                page = chunk.metadata.get('page', '')
                writer.writerow([i, page, chunk.page_content])
        print(f"Saved OCR chunks to {csv_path}")

        # Initialize embeddings with OpenAI API key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create FAISS vector store from document chunks and embeddings
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        print('Saving FAISS vector store to disk...')
        vector_store.save_local(faiss_path)

    # Return the retriever for document retrieval with specified search_type
    retriever = vector_store.as_retriever(
        search_type="similarity",  # or "mmr" or "similarity_score_threshold"
        search_kwargs={"k": 100}  # Adjust the number of results if needed
    )
    return retriever

# Function to get the response from the RAG system
async def get_rag_response(query: str):
    retriever = setup_rag_system()

    # Retrieve the relevant documents using 'get_relevant_documents' method
    retrieved_docs = retriever.invoke(query)

    # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # LLM expects a list of strings (prompts), so we create one by combining the query with the retrieved context
    prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

    # Generate the final response using the language model (LLM)
    generated_response = llm.generate(prompt)  # Pass as a list of strings
    
    return generated_response













