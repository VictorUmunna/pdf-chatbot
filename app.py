import streamlit as st
from pdfchat import PDFChatbot

def main():

    st.set_page_config("Chat with your file")
    st.header("Chat with your File 🗣️")

    pdf_chatbot = PDFChatbot()

    #Uploading the PDF Files:
    with st.sidebar:
        st.title("file Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing...."):
                raw_text = pdf_chatbot.xtract_pdf_text(pdf_docs)
                text_chunks = pdf_chatbot.get_text_chunks(raw_text)
                pdf_chatbot.save_embedding_vector_store(text_chunks)
                st.success("Done")

    #Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Please ask a question from your uploaded files?"}]

    #Displaying chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Getting the user's input
    user_question = st.chat_input("Ask a Question from your PDF Files")

    #Checking if the user has input a message
    if user_question:
        #Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)

        #Adding the user question to chat history
        st.session_state.messages.append({"role":"user","content":user_question})

        #Getting the respoonse
        response = pdf_chatbot.user_input(user_question)
        #Displyaing the assistant response

        with st.chat_message("assistant"):
            st.markdown(response)

        #Adding the assistant response to chat history
        st.session_state.messages.append({"role":"assistant","content":response})

    #Function to clear history
    def clear_chat_history():
        st.session_state.messages = [{"role":"assistant","content":"Please ask a question from your uploaded files?"}]

    #Button for clearing history
    st.sidebar.button("Clear Chat History",on_click=clear_chat_history)  

    if __name__ == "__main__":
        main()