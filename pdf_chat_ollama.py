    import streamlit as st
    import os
    import logging
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain_community.llms import Ollama

    # App configuration
    st.set_page_config(page_title="PDF Chat with Local DeepSeek-R1 1.5B")
    st.title("💬 Chat with PDF using Local DeepSeek-R1 1.5B")
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("********** Initialize embeddings and text splitter ****************")
    def initialize_components():
        """Initialize embeddings and text splitter"""
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), \
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    logging.info("********** Process PDF and create vector store ****************")
    def process_pdf(file_path):
        """Process PDF and create vector store"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split documents
       # logging.info("********** Split documents ****************")
        texts = st.session_state.text_splitter.split_documents(documents)

        # Create vector store
        logging.info("********** Create vector store ****************")
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=st.session_state.embeddings,
            persist_directory="chroma_db"
        )
        return vector_store

    logging.info("********** Initialize embeddings and text splitter ****************")
    # Initialize embeddings and text splitter
    if "embeddings" not in st.session_state or "text_splitter" not in st.session_state:
        st.session_state.embeddings, st.session_state.text_splitter = initialize_components()

    # PDF Upload
    logging.info("********** PDF Upload ****************")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # Process PDF when file is uploaded
    logging.info("********** Process PDF when file is uploaded ****************")
    if uploaded_file and ("vector_store" not in st.session_state):
        with st.spinner("Processing PDF..."):
            # Save temporary file
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process PDF
            st.session_state.vector_store = process_pdf(temp_file)
            st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            os.remove(temp_file)
        st.success("PDF processed successfully!")
    logging.info("********** PDF processed successfully! ****************")

    # Initialize Ollama LLM

    if "llm" not in st.session_state:
        logging.info("********** Initialize Ollama LLM DeepSeek-R1 1.5B ****************")
        with st.spinner("Initializing DeepSeek-R1 1.5B..."):
            st.session_state.llm = Ollama(
                model="deepseek-r1:1.5b",  # Ensure this matches your Ollama model name
                temperature=1.3,
                num_ctx=2048,
                verbose=True
            )

    logging.info("********** Initialize chat history  ****************")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the PDF"):
        if "vector_store" not in st.session_state:
            st.error("Please upload a PDF first!")
            st.stop()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            verbose=True,
            return_source_documents=True
        )

        # Get response
        with st.spinner("Thinking..."):
            try:
                result = qa_chain(prompt)
                response = f"{result['result']}\n\nSources: {list(set([doc.metadata['source'] for doc in result['source_documents']]))}"
            except Exception as e:
                response = f"Error: {str(e)}"

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)