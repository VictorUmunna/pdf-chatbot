#Chat PDF Script
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma #vector embeddngs
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from itertools import count


load_dotenv()

#configuring the api key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class mcq_generation():
    def __init__(self):
        pass

    #Extract the text from the PDF 
    def extract_pdf_text(pdf_file):
        text=""
        for pdf in pdf_file:
            pdf_reader= PdfReader(pdf)
            # reads all the pages in the pdf and extracting the texts from the pdf
            for page in pdf_reader.pages:
                text+= page.extract_text()
        return  text

    def save_embedding_vector_store(text_chunks):
        embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        db =Chroma.from_texts(text_chunks,embedding=embeddings, persist_directory="./chroma_db")
        db.persist()

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n{context}?\nQuestion:\n{question}\n\nAnswer:\n
        """

        # Initialize the conversational AI model
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

        # Create a prompt template object
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Load the QA chain for generating responses
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain
    
    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        #load the indexes from the vector database
        new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, allow_dangerous_deserialization=True)
        #check the similarity with the user question
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)
        
        return response["output_text"]
