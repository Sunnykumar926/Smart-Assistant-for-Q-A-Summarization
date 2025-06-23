import os
import time
import random
import hashlib
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PyPDF2.errors import PdfReadError
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

import streamlit as st
from streamlit_option_menu import option_menu 
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(files):
    documents = []
    for file in files:
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        # Store filename only without path
                        filename = os.path.basename(file.name)
                        metadata = {"source": filename, "page": page_num+1}
                        documents.append(Document(page_content=text, metadata=metadata))
            
            elif file.name.endswith('.txt'):
                file.seek(0)
                text = file.read().decode('utf-8')
                filename = os.path.basename(file.name)
                metadata = {"source": filename, "page": 1}
                documents.append(Document(page_content=text, metadata=metadata))
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            
    return documents

def generate_summary(docs):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    summaries = {}
    
    # Group pages by document
    doc_pages = {}
    for doc in docs:
        filename = doc.metadata["source"]
        if filename not in doc_pages:
            doc_pages[filename] = []
        doc_pages[filename].append(doc.page_content)
    
    # Generate summary for each document
    for filename, pages in doc_pages.items():
        combined_content = "\n\n".join(pages)[:15000]  # Limit to 15k chars
        
        prompt = f"""
        Generate a concise summary of the document in 150 words or fewer. 
        Focus exclusively on main points and key findings.

        Document Content:
        {combined_content}

        Return only the summary without any headings.
        """
        
        try:
            response = model.invoke(prompt)
            summaries[filename] = response.content.strip()
        except Exception as e:
            st.error(f"Summary generation failed for {filename}: {e}")
            summaries[filename] = "Summary unavailable due to an error."
    
    return summaries


def evaluate_answer(model, context, question, user_answer, source):
    prompt = f"""
    Context: {context}

    Question: {question}
    User Answer: {user_answer}

    **Evaluation Criteria:**
    1. Only use the provided context to evaluate the answer. Do not use any outside knowledge.
    2. Assess the user's answer based on:
        - ✅ Accuracy (Is the answer factually correct according to the context?)
        - ✅ Completeness (Does it cover all key points?)
        - ✅ Document Grounding (Are details supported directly by the text?)
    **Your Response Should Include:**
    - A clear verdict: Correct / Partially correct / Incorrect
    - A brief justification using evidence from the context
    - If incorrect or partially correct, provide the correct answer using only the context

    Be helpful and constructive.
    """

    response = model.invoke(prompt)
    st.write("💡 Feedback:")
    st.write(response.content.strip())
    
    # Show the specific source for this question
    # st.caption(f"📚 Source: {source}")


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000, 
        chunk_overlap=1000,
        separators=['\n\n', '\n', '.', ' ']
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    **Strict Instructions:**
    1. Answer the question as detailed as possible from the provided context ONLY
    2. If unsure, say "I cannot find this in the documents"
    3. Never speculate or invent details
    4. Consider our conversation history when answering

    Conversation History:
    {chat_history}

    Context:\n {context}?\n
    Question: \n{question}\n

    **Reasoning Steps:**
    1. Identify relevent context passages
    2. Extract verbatim quotes supporting the answer
    3. If no evidence exists, state so explicitly

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'  # Set output_key here instead
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
    )
    return qa_chain


def user_input(user_question):
    chain = get_conversational_chain()
    
    response = chain(
        {"question": user_question},
        return_only_outputs=True
    )
    
    output_text = response['answer']
    source_docs = response.get('source_documents', [])
    
    # Check if the model couldn't find the answer in the documents
    not_found_phrases = [
        "cannot find",
        "not in the documents",
        "not found",
        "i don't know",
        "no information"
    ]
    
    answer_not_found = any(phrase in output_text.lower() for phrase in not_found_phrases)
    
    # Format the response
    st.markdown(f"💬 **Reply:** \n\n{output_text}")

    # Show sources from the response
    if source_docs and not answer_not_found:
        used_sources = set()
        for doc in source_docs:
            source_info = doc.metadata["source"]
            used_sources.add(source_info)
        st.markdown("📚 **Source(s):** " + ", ".join(sorted(used_sources)))
    else:
        st.markdown("ℹ️ No specific sources referenced in this answer")

# -------------------- PAGE CONFIG -------------

st.set_page_config(page_title='Smart Assistant', page_icon='kfgksgf🤖', layout='wide')
if 'current_docs_hash' not in st.session_state:
    st.session_state.current_docs_hash = None\

if 'memory_initialized' not in st.session_state:
    st.session_state.memory_initialized=False 

# -------------------- MENU BAR -----------------

with st.container():
    selected = option_menu(
        menu_title=None,
        options = ['About', 'Question Answering', 'Check Knowledge'],
        icons = ['info-circle','chat-left-text','lightning-charge'],
        orientation='horizontal',
        default_index=0,
        styles={
            'container': {'padding': '0!important'}, 
            'icon' : {'color':'#9A341', 'font-size':'16px'},
            'nav-link': {'font-size': '16px', 'text-align':'center', 'margin':'5px', '--hover-color':'#FFA07A'},
            'nav-link-selected': {'background-color': '#2c7be5'}
        }
    )

# ------------------- ABOUT ---------------------

if selected == 'About':
    st.markdown("## 📄 Document-Aware GenAI Assistant")
    st.markdown("""
    **Unlock the power of your documents** with our intelligent assistant that understands, reasons, and explains content like never before. 
    Go beyond simple keyword searches with AI that comprehends context, relationships, and logical implications within your documents.
                
    ### 🧠 Core Capabilities:
    - **Deep Document Understanding**: Extract meaning from complex PDFs and text files  
    - **Contextual Question Answering**: Get precise answers with source references  
    - **Intelligent Knowledge Checks**: Automatically generate comprehension questions  
    - **Critical Thinking Enhancement**: Evaluate answers against document content  
    - **Multi-document Analysis**: Connect insights across multiple files  
                
    ### ⚙️ How It Works:
    1. **Upload** your PDFs or text documents  
    2. **Process** to build an AI-understandable knowledge base  
    3. **Ask questions** in natural language  
    4. **Test knowledge** with automatically generated challenges  
    5. **Receive feedback** with document-grounded explanations  
                
    *Built with Google Gemini AI for advanced reasoning and Streamlit for intuitive interaction*

    ---
    © 2025 Sunny Kumar. All rights reserved.
    """)

    


if selected == 'Question Answering':
    st.title("🔍 Document Processing Center")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['pdf', 'txt'])

    if uploaded_files:
        # Create hash of current files to detect changes
        files_hash = hashlib.md5(",".join(sorted([f.name for f in uploaded_files])).encode()).hexdigest()

        # Reset state if documents change
        if files_hash != st.session_state.get('current_docs_hash'):
            # Clear previous document state
            keys_to_clear = ['docs_processed', 'challenge_questions', 
                            'challenge_context', 'challenge_docs',
                            'generation_count', 'current_docs_hash']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state.current_docs_hash = files_hash
        
        st.session_state.pdf_docs = uploaded_files

    if st.button("Process Documents"):
        if "pdf_docs" in st.session_state:
            with st.spinner("Processing..."): 
                raw_text = get_pdf_text(st.session_state.pdf_docs) 

                summaries = generate_summary(raw_text)
                st.session_state.summary = summaries

                text_chunks = get_text_chunks(raw_text) 
                get_vector_store(text_chunks) 
                st.session_state.docs_processed = True
                st.session_state.memory_initialized=False
                st.success("Documents processed successfully!")

                # Display each summary with document name heading
                for filename, summary in summaries.items():
                    with st.expander(f"📘 Summary: {filename}", expanded=True):
                        st.write(summary)
                
                # Reset challenge state for new documents
                if 'challenge_questions' in st.session_state:
                    del st.session_state.challenge_questions
                if 'challenge_context' in st.session_state:
                    del st.session_state.challenge_context
                if 'challenge_docs' in st.session_state:
                    del st.session_state.challenge_docs
                if 'generation_count' in st.session_state:
                    st.session_state.generation_count = 0

    
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.header("Documents Query Interface")

    user_question = st.text_input("Ask Questions from the uploaded documents .. ✍️📝")

    if user_question:
        user_input(user_question)

if selected == 'Check Knowledge':
    st.title("🧠 Check your knowledge")

    # Initialize session state keys
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = None
    if 'challenge_contexts' not in st.session_state:
        st.session_state.challenge_contexts = None
    if 'challenge_sources' not in st.session_state:
        st.session_state.challenge_sources = None
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []  # Store all document chunks
    if 'used_chunks' not in st.session_state:
        st.session_state.used_chunks = set()  # Track which chunks have been used

    if "docs_processed" not in st.session_state:
        st.warning("Please upload and process documents in the 'Question Answering' section first.")
    else:
        # Initialize chunks if not already done
        if not st.session_state.all_chunks:
            raw_documents = get_pdf_text(st.session_state.pdf_docs)
            text_chunks = get_text_chunks(raw_documents)
            st.session_state.all_chunks = text_chunks
            st.session_state.used_chunks = set()

        # Always show the generate button
        if st.button("Generate New Questions"):
            with st.spinner("Creating new questions from your documents..."):
                # Get available chunks (not used yet)
                available_chunks = [
                    chunk for idx, chunk in enumerate(st.session_state.all_chunks)
                    if idx not in st.session_state.used_chunks
                ]
                
                # If not enough available, reset used chunks
                if len(available_chunks) < 3:
                    st.info("Reusing document content to create new questions")
                    st.session_state.used_chunks = set()
                    available_chunks = st.session_state.all_chunks
                
                # Randomly select 3 chunks from available chunks
                selected_chunks = random.sample(available_chunks, min(3, len(available_chunks)))
                
                # Mark selected chunks as used
                for chunk in selected_chunks:
                    chunk_idx = st.session_state.all_chunks.index(chunk)
                    st.session_state.used_chunks.add(chunk_idx)
                
                contexts = []
                sources_used = set()
                for chunk in selected_chunks:
                    source = chunk.metadata.get('source', 'unknown')
                    filename = os.path.basename(source)
                    contexts.append({
                        "source": filename,
                        "content": chunk.page_content[:2000]
                    })
                    sources_used.add(filename)
                
                st.session_state.challenge_contexts = contexts
                st.session_state.challenge_sources = sources_used
                
                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
                
                # Generate questions individually for each context
                questions = []
                for context in contexts:
                    prompt = f"""
                    Based on the following document content, generate one logic-based or comprehension-focused question:
                    
                    Document Source: {context['source']}
                    Content:
                    {context['content']}
                    
                    Return ONLY the question without any numbering or additional text.
                    """
                    
                    try:
                        response = model.invoke(prompt)
                        question = response.content.strip()
                        # Remove numbering if present
                        if question and question[0].isdigit():
                            question = question.split(".", 1)[-1].strip()
                        questions.append(question)
                    except Exception as e:
                        st.error(f"Failed to generate question: {str(e)}")
                        questions.append("Question generation failed")
                
                st.session_state.challenge_questions = questions
                st.session_state.generation_count += 1

        # Display existing questions if available
        if st.session_state.challenge_questions:
            st.subheader(f"Question Set #{st.session_state.generation_count}")
            if st.session_state.get('challenge_sources'):
                st.caption(f"Answer these questions based on your documents")
            
            for i, (q, ctx) in enumerate(zip(
                st.session_state.challenge_questions,
                st.session_state.challenge_contexts
            ), 1):
                st.markdown(f"**Q{i}: {q}**")
                st.caption(f"Source: {ctx['source']}")
                
                user_answer = st.text_area(
                    f"Your answer for Q{i}:",
                    key=f"user_answer_{i}_{st.session_state.generation_count}",
                    height=100
                )
                
                if user_answer:
                    evaluate_answer(
                        model=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3),
                        context=ctx['content'],
                        question=q,
                        user_answer=user_answer,
                        source=ctx['source']
                    )