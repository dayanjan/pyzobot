
import streamlit as st  # type: ignore

import os
import shutil
import uuid
import re
import tempfile
from pyzotero import zotero
import pandas as pd
import requests

from llama_index.core import download_loader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Initialize variables at the top
split_docs = None
embeddings = None
ids = None
if "k" not in st.session_state:
    st.session_state["k"] = 5  # Default value
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 1000  # Default value
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gpt-4"  # Default value


st.set_page_config(page_title="PyZoBot: Zotero Library QA System", layout="wide" )


header_html = """
<div style="text-align: center;">
    <img src='https://i.postimg.cc/4xPdhkB2/PYZo-Bot-new-logo-small.png' alt='PyZoBot Logo' style='width:auto; height:40%;'>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Display the title within the app
# st.title("PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation")
st.markdown(
    "<h1 style='text-align: center;'>PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation</h1>",
    unsafe_allow_html=True
)
st.markdown("---") 


# User inputs
with st.sidebar:
    openai_api_key = st.text_input(
        "**Enter your OpenAI API key:**", type="password", key="openai_api_key"
    )
    zotero_api_key = st.text_input(
        "**Enter your Zotero API key:**", type="password", key="zotero_api_key"
    )
    library_type = st.selectbox(
        "**Select your Zotero library type:**", ["group", "user"], key="library_type"
    )
    library_id = st.text_input("**Enter your Zotero library ID:**", key="library_id")
    chunk_size = st.number_input(
        "**Enter chunk size:**", min_value=100, max_value=5000, value=500, key="chunk_size"
    )
    chunk_overlap = st.number_input(
        "**Enter chunk overlap:**",
        min_value=0,
        max_value=5000,
        value=200,
        key="chunk_overlap",
    )
    st.session_state["model_name"] = st.selectbox(
        "**Select OpenAI model:**", 
        ["gpt-4", "gpt-3.5-turbo"], 
        index=["gpt-4", "gpt-3.5-turbo"].index(st.session_state["model_name"]), 
        key="model_name_select"
    )
    st.session_state["max_tokens"] = st.number_input(
        "**Enter max tokens:**", 
        min_value=100, 
        max_value=4000, 
        value=st.session_state["max_tokens"], 
        key="max_tokens_input"
    )
    st.session_state["k"] = st.slider(
        "**Number of documents to retrieve:**", 
        min_value=1, 
        max_value=30, 
        value=st.session_state["k"], 
        key="k_slider"
    )
     
    st.sidebar.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
    # Add a button to apply new chunking parameters
    st.sidebar.markdown("""
    <h3 style='color: red;'><strong>Apply New Chunking Parameters</strong></h3>
                        
    **Press this button to update how your documents are processed after changing any of the following settings:**
                        
    Relevant parameters:
    1. Chunk size: Controls the length of text segments.
    2. Chunk overlap: Determines how much text is shared between adjacent chunks.
    """, unsafe_allow_html=True)
    if st.button("Apply New Chunking Parameters"):
        st.session_state["apply_new_chunking"] = True
    else:
        st.session_state["apply_new_chunking"] = False


# Function to handle library ID change
def handle_library_id_change():
    if "previous_library_id" in st.session_state:
        if library_id != st.session_state["previous_library_id"]:
            # Library ID has changed
            st.session_state["library_id_changed"] = True
            # Remove 'db' and related data from session state if they exist
            for key in ["db", "all_documents", "pdf_paths", "chroma_persist_dir"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Remove temporary directories
            if "temp_dir" in st.session_state:
                shutil.rmtree(st.session_state["temp_dir"], ignore_errors=True)
                del st.session_state["temp_dir"]
            if "chroma_persist_dir" in st.session_state:
                shutil.rmtree(st.session_state["chroma_persist_dir"], ignore_errors=True)
                del st.session_state["chroma_persist_dir"]
    else:
        st.session_state["library_id_changed"] = True  # First time
    # Update previous_library_id
    st.session_state["previous_library_id"] = library_id

# Call the function to handle library ID change
handle_library_id_change()



def create_unique_temp_dir():
    # Remove the existing directory if it exists
    if "temp_dir" in st.session_state:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    # Create a new temporary directory
    st.session_state.temp_dir = tempfile.mkdtemp()
    return st.session_state.temp_dir

def process_zotero_library(library_id, zotero_api_key, library_type):
    zot = zotero.Zotero(
        library_id=library_id, library_type=library_type, api_key=zotero_api_key
    )
    items = zot.everything(zot.top())

    # Process Zotero items
    df = pd.json_normalize(items)
    df1 = df[df["meta.numChildren"] == 0]
    df2 = df[df["meta.numChildren"] != 0]
    df2["links.self.href"] = df2["links.self.href"].astype(str) + "/children"
    frames = [df1, df2]
    df3 = pd.concat(frames)
    df4 = df3

    def fetch_url_content_as_json(url):
        try:
            headers = {"Zotero-API-Key": f"{zotero_api_key}"}  # Adjust the header
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()  # Parse JSON response
            else:
                return {"error": f"Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

    df4["JSONContent"] = df4["links.self.href"].apply(fetch_url_content_as_json)

    def flatten_json(nested_json: dict, exclude: list = [""]) -> dict:
        """
        Flatten a list of nested dicts.
        """
        out = dict()

        def flatten(x: (list, dict, str), name: str = "", exclude=exclude):
            if type(x) is dict:
                for a in x:
                    if a not in exclude:
                        flatten(x[a], f"{name}{a}.")
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, f"{name}{i}.")
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(nested_json)
        return out

    df_source2 = pd.DataFrame([flatten_json(x) for x in df4["JSONContent"]])

    df9 = df_source2
    cols_to_join = [col for col in df9.columns if col.endswith(".enclosure.href")]
    df9["enclosure.href"] = df9[cols_to_join].apply(
        lambda x: "##".join(x.values.astype(str)), axis=1
    )

    df10 = df9
    cols_to_join = [col for col in df10.columns if col.endswith(".enclosure.title")]
    df10["enclosure.title"] = df10[cols_to_join].apply(
        lambda x: "##".join(x.values.astype(str)), axis=1
    )
    df11 = df10[["enclosure.title", "enclosure.href"]]
    df12 = df11
    new_df = (
        df12["enclosure.title"]
        .str.split("##", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("enclosure.title")
    )
    df12 = df12.drop("enclosure.title", axis=1).join(new_df)
    df13 = df12
    new_df2 = (
        df13["enclosure.href"]
        .str.split("##", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("enclosure.href")
    )
    df13 = df13.drop("enclosure.href", axis=1).join(new_df2)
    df13.dropna(inplace=True)
    df15 = df13
    df15 = df15.replace("nan", pd.NA)
    df15 = df15.dropna()
    df15["PDF_Names"] = df15["enclosure.title"]
    df15 = df15[["PDF_Names", "enclosure.href"]]
    df16 = df15.drop_duplicates(keep="first")
    df17 = df16[df16["PDF_Names"].str.endswith(".pdf")]
    df20 = df17
    # Define your output folder
    output_folder = create_unique_temp_dir()
    headers = {"Zotero-API-Key": f"{zotero_api_key}"}
    # Iterate through the dataframe
    for index, row in df20.iterrows():
        api_url = row["enclosure.href"]
        pdf_filename = row["PDF_Names"]
        # Make an HTTP GET request for each URL
        response = requests.get(api_url, headers=headers)
        # Check if the request was successful
        if response.status_code == 200:
            binary_content = response.content
            content_type = response.headers.get("Content-Type")
            # Check if the content type is 'application/pdf'
            if content_type == "application/pdf":
                pdf_filename = row["PDF_Names"]
                pdf_filepath = os.path.join(output_folder, pdf_filename)
                # Save the PDF to the specified folder
                with open(pdf_filepath, "wb") as pdf_file:
                    pdf_file.write(binary_content)
                print(f"Saved PDF: {pdf_filename}")
            else:
                print(f"Skipped non-PDF content for URL: {api_url}")
        else:
            print(f"Failed to fetch data from the API for URL: {api_url}")
    return output_folder

if st.button("Fetch PDFs from Zotero"):
    if zotero_api_key and library_id and library_type:
        try:
            with st.spinner("Fetching PDFs..."):
                temp_dir = process_zotero_library(
                    library_id, zotero_api_key, library_type
                )
                st.success(f"PDFs saved in temporary directory: {temp_dir}")
                # List of PDF paths
                pdf_paths = []
                # Get the full paths of the saved PDF files
                pdf_files = os.listdir(temp_dir)
                if pdf_files:
                    st.write("Saved PDFs:")
                    for file in pdf_files:
                        st.write(file)
                        # Add full path to pdf_paths list
                        full_path = os.path.join(temp_dir, file).replace("\\", "//")
                        pdf_paths.append(full_path)
                    # Store pdf_paths in session state
                    st.session_state["pdf_paths"] = pdf_paths
                else:
                    st.write("No PDF files saved.")
            # Set library_id_changed to True to trigger processing
            st.session_state["library_id_changed"] = True
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter all required Zotero details.")

# Import the necessary loader
PyMuPDFReader = download_loader("PyMuPDFReader")
# Initialize the loader
loader = PyMuPDFReader()
# A list to store all processed documents
all_documents = []

# Check if PDFs have been processed
if (
    "pdf_paths" in st.session_state
    and st.session_state["pdf_paths"]
    and st.session_state.get("library_id_changed", False)
):
    try:
        with st.spinner("Document loading..."):
            # Process each PDF file
            for pdf_file in st.session_state["pdf_paths"]:
                documents = loader.load_data(file_path=pdf_file, metadata=True)
                # Extend the list with documents from the current file
                all_documents.extend(documents)
            # Store documents in session state
            st.session_state["all_documents"] = all_documents
            if all_documents:
                st.success("Documents processed successfully.")
            else:
                st.write("No documents were loaded.")
        # Set library_id_changed to True to trigger vector store building
        st.session_state["library_id_changed"] = True
    except Exception as e:
        st.error(f"An error occurred while processing the PDFs: {str(e)}")
elif "all_documents" in st.session_state:
    all_documents = st.session_state["all_documents"]
else:
    st.write("No PDFs found in session state.")


# Build vector store if needed
if openai_api_key and all_documents:
    # Build vector store if not already built or if library_id changed or new chunking params are applied
    if ("db" not in st.session_state or 
        st.session_state.get("library_id_changed", False) or 
        st.session_state.get("apply_new_chunking", False)):
        try:
            # Start the spinner for the chunking process
            with st.spinner("Chunking documents..."):
                # Initialize the OpenAI embeddings
                embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key, model="text-embedding-ada-002"
                )
                # Set chunking parameters
                chunk_size_limit = chunk_size
                max_chunk_overlap = chunk_overlap
                # Initialize the text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap
                )
                split_docs = []
                # Chunk each document
                for document in all_documents:
                    text = document.get_text()  # Get the document text
                    source = document.metadata["file_path"].split("/")[
                        -1
                    ]  # Extract file name as source
                    # Split the text into chunks
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        metadata = {"source": source}
                        chunk_instance = Document(page_content=chunk, metadata=metadata)
                        split_docs.append(chunk_instance)
                # Create ids for chunks if they exist
                if split_docs:
                    ids = [str(i) for i in range(1, len(split_docs) + 1)]
                # Display success message after successful chunking
                if split_docs and ids:
                    st.success(f"{len(split_docs)} chunks were successfully created.")
                else:
                    st.write("No chunks were created.")
            
            # Start the spinner for the vectorizing and saving process
            with st.spinner("Building and saving vector store..."):
                # Build the Chroma vector store and save it to disk
                st.session_state["chroma_persist_dir"] = tempfile.mkdtemp()
                persist_directory = st.session_state["chroma_persist_dir"]
                # Sanitize library_id
                library_id_sanitized = re.sub(r"\W+", "_", library_id)
                # Generate a random collection ID using the uuid library
                collection_id = str(uuid.uuid4())[:8]
                # Combine library_id with the random collection_id
                collection_name = f"user_vectors_{library_id_sanitized}_{collection_id}"
                # Ensure the collection name adheres to Chroma's requirements
                if len(collection_name) > 63:
                    collection_name = collection_name[:63]
                if not collection_name[0].isalnum():
                    collection_name = "a" + collection_name[1:]
                if not collection_name[-1].isalnum():
                    collection_name = collection_name[:-1] + "z"
                # Build the Chroma vector store and save it to disk
                db = Chroma.from_documents(
                    split_docs,
                    embeddings,
                    collection_name=collection_name,
                    ids=ids,
                    persist_directory=persist_directory,
                )
                db.persist()
                st.success("Built new vector store and saved it to disk.")
            
            # Store the db in session state
            st.session_state["db"] = db
            # Set library_id_changed and apply_new_chunking to False
            st.session_state["library_id_changed"] = False
            st.session_state["apply_new_chunking"] = False
        except KeyError as e:
            st.error(f"Chunking process failed: {str(e)}")
    else:
        db = st.session_state["db"]
else:
    st.error("Please enter your OpenAI API key and ensure documents are processed.")

##################################################################################
import io

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def answer_question(question, db):
    system_template = """
        Answer the user query using ONLY the relevant content provided in this prompt.
        Do not use any external knowledge or information not present in the given context.
        If the provided content does not contain information to answer the query, respond with "I don't have enough information to answer this question based on the given context."
        Take your time and provide as much information as you can in the answer.\n

        For each statement in your answer, provide in-text citations after the sentence, e.g., [1].
        Start with number [1] every time you generate an answer and make the number matching the source document.
        If a statement has multiple citations, provide them all, e.g., [1], [2], [3].

        By the end of the answer, provide a References section as Markdown (### References) including the number and the file name, e.g.:
        [1] Author et al. - YEAR - file name.pdf

        Write each reference on a new line.

        {summaries}
        """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(
        model_name=st.session_state["model_name"],
        temperature=0,
        max_tokens=st.session_state["max_tokens"],
        openai_api_key=openai_api_key,
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type="mmr", search_kwargs={"k": st.session_state["k"], "lambda_mult": 0.25}
        ),
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )

    result = chain(question)
    return result

# Function to display chat history
def display_chat_history():
    for i, (q, a, sources) in enumerate(st.session_state["chat_history"]):
        st.write(f"**Question {i+1}:** {q}")
        st.write(f"**Answer:** {a}")
        if sources:
            st.write(f"**Source Documents:**")
            for index, doc in enumerate(sources, start=1):
                st.write(f"{index}: {doc}")

# Chat input at the bottom of the interface


with st.container():
    st.subheader("Chat History")
    # Display chat history
    if st.session_state["chat_history"]:
        display_chat_history()
    # Input question and get answer button
    st.markdown("<hr style='border: 2px solid #333;'>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Target the label for the question input */
    .question-input-label {
        font-size: 20px;
        color: #ff3633;
        margin-bottom: -30px;  /* Reduce space below the label */
    }
    
    /* Target the specific text input field with a unique key */
    div[data-testid="stTextInput"][aria-describedby="question_input"] input {
        font-size: 20px;  /* Adjust input text size */
        color: #ff3633;  /* Set custom text color */
    }
    </style>
    <p class="question-input-label"><strong>Enter your question:</strong></p>
    """, unsafe_allow_html=True)
    question = st.text_input("", key="question_input")

    
    if st.button("Get Answer"):
        if "db" not in st.session_state:
            st.error("Please process the Zotero library first.")
        else:
            with st.spinner("Generating answer..."):
                result = answer_question(question, st.session_state["db"])
            
            answer = result["answer"]
            sources = result["source_documents"]
            # Save the question, answer, and sources in the session state chat history
            st.session_state["chat_history"].append((question, answer, sources))
            st.subheader("Answer:")
            st.write(answer)
            st.markdown("---") 
            if sources:
                st.subheader("Source Documents:")
                for index, doc in enumerate(sources, start=1):
                    st.write(f"{index}: {doc}")
            st.subheader("All relevant sources:")
            for source in set([doc.metadata["source"] for doc in sources]):
                st.write(source)
                
    # Prepare chat history for download
    if st.session_state["chat_history"]:
        chat_history_str = ""
        for i, (q, a, sources) in enumerate(st.session_state["chat_history"]):
            chat_history_str += f"Question {i+1}: {q}\n"
            chat_history_str += f"Answer: {a}\n"
            chat_history_str += f"Source Documents:\n"
            for index, doc in enumerate(sources, start=1):
                chat_history_str += f"{index}: {doc}\n"
            chat_history_str += "\n"
        # Convert the chat history to a downloadable format (txt)
        chat_history_bytes = io.BytesIO(chat_history_str.encode("utf-8"))
        # Display the download button
        st.download_button(
            label="Download Chat History",
            data=chat_history_bytes,
            file_name="chat_history.txt",
            mime="text/plain",
        )

