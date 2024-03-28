from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

# Configure the Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class PDFChatbot:
    """A chatbot designed to interact with content from PDF documents."""
    
    def __init__(self):
        pass

    @staticmethod
    def extract_pdf_text(pdf_file):
        """Extract text from PDF files.

        Args:
            pdf_file (str or list): The path to the PDF file or a list of paths.

        Returns:
            str: Extracted text from the PDF files.
        """
        text = ""
        for pdf in pdf_file:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @staticmethod
    def get_text_chunks(text):
        """Split text into smaller chunks.

        Args:
            text (str): The text to be split.

        Returns:
            list: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    
    @staticmethod
    def save_embedding_vector_store(text_chunks):
        """Save embedding vector store for text chunks.

        Args:
            text_chunks (list): List of text chunks.
        """
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        db = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
        db.persist()

    @staticmethod
    def get_conversational_chain():
        """Get a conversational AI chain for generating responses.

        Returns:
            object: The conversational AI chain.
        """
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n{context}?\nQuestion:\n{question}\n\nAnswer:\n
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    
    @staticmethod
    def user_input(user_question):
        """Provide response to a user question.

        Args:
            user_question (str): The question asked by the user.

        Returns:
            str: The response to the user question.
        """
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = PDFChatbot.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
