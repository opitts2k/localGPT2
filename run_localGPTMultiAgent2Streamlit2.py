import os
import logging
import asyncio
import streamlit as st
import torch
import sys
import requests
import json
import time
import re
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Mapping, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Disable Streamlit's file watcher to avoid conflicts with PyTorch
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Import the MultiAgentLLM class from run_localGPTMultiAgent2.py
try:
    # We'll use subprocess to run the LLM to avoid import conflicts
    import subprocess
    
    # Check if the run_localGPTMultiAgent2.py file exists
    if not os.path.exists("run_localGPTMultiAgent2.py"):
        logger.error("run_localGPTMultiAgent2.py file not found")
        st.error("run_localGPTMultiAgent2.py file not found")
    else:
        logger.info("Found run_localGPTMultiAgent2.py file")
        
    # Import torch separately for system information
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    st.error(f"Error importing required modules: {e}")

# Shared Constants
OLLAMA_HOST = "http://localhost:11434"  # Ollama server base URL
INFERENCE_SERVER_URL = f"{OLLAMA_HOST}/api/chat"  # Ollama chat API endpoint
EMBEDDING_SERVER_URL = f"{OLLAMA_HOST}/api/embeddings"  # Ollama embeddings endpoint
MODEL_NAME = "llama3:8b"  # Default model
EMBEDDING_MODEL_NAME = None  # Will use the same as MODEL_NAME by default
PDF_DIRECTORY = "./pdfs"
PERSIST_DIRECTORY = "./DB"  # Use local DB directory
RESPONSES_DIRECTORY = "./llmresponses"  # Directory to save LLM responses

class Config:
    """Configuration management"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    
    @classmethod
    def validate(cls):
        missing = [k for k, v in cls.__dict__.items() 
                  if not v and not k.startswith('_') and k.endswith('_KEY')]
        if missing:
            logger.warning(f"Missing environment variables: {', '.join(missing)}")
            return missing
        return []

class ServiceError(Exception):
    """Custom exception for service errors"""
    def __init__(self, service: str, message: str, original_error: Exception = None):
        self.service = service
        self.message = message
        self.original_error = original_error
        super().__init__(f"{service}: {message}")

def get_available_ollama_models():
    """
    Get a list of available Ollama models.
    Returns a list of dictionaries with model information.
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            logger.error(f"Error fetching models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return []

def pull_ollama_model(model_name, progress_callback=None):
    """
    Pull an Ollama model from the repository.
    
    Args:
        model_name: Name of the model to pull
        progress_callback: Optional callback function to report progress
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Pull the model
        pull_url = f"{OLLAMA_HOST}/api/pull"
        pull_data = {"name": model_name}
        
        response = requests.post(pull_url, json=pull_data, stream=True)
        response.raise_for_status()
        
        # Process the streaming response to show progress
        for line in response.iter_lines():
            if line:
                progress_data = json.loads(line)
                if "status" in progress_data:
                    if progress_callback:
                        if progress_data.get("completed"):
                            progress_callback(f"Model pull completed: {progress_data.get('status')}", 100)
                        else:
                            progress = progress_data.get("progress", 0)
                            progress_callback(f"Pulling model: {progress_data.get('status')}", progress)
                    
                    if progress_data.get("completed"):
                        logger.info(f"Model pull completed: {progress_data.get('status')}")
                    else:
                        logger.info(f"Pulling model: {progress_data.get('status')} - {progress_data.get('progress', 0)}%")
        
        logger.info(f"Successfully pulled model {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        if progress_callback:
            progress_callback(f"Error pulling model: {str(e)}", 0)
        return False

def ensure_ollama_model(model_name, progress_callback=None):
    """
    Ensure the specified Ollama model is available.
    If not, pull it from Ollama's model repository.
    """
    logger.info(f"Checking if model {model_name} is available in Ollama...")
    
    # Check if model exists
    try:
        models = get_available_ollama_models()
        
        # Check if our model is in the list
        model_exists = any(model.get("name") == model_name for model in models)
        
        if not model_exists:
            logger.info(f"Model {model_name} not found. Pulling from Ollama repository...")
            return pull_ollama_model(model_name, progress_callback)
        else:
            logger.info(f"Model {model_name} is already available")
            if progress_callback:
                progress_callback(f"Model {model_name} is already available", 100)
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error ensuring Ollama model: {e}")
        logger.error("Make sure Ollama server is running at: " + OLLAMA_HOST)
        if progress_callback:
            progress_callback(f"Error: {str(e)}", 0)
        return False

def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file"""
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    pdf_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return pdf_path

def ingest_pdfs(files):
    """Ingest PDF files into the vector database"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        st.error("Required packages not installed. Please install langchain, langchain_community, and PyPDF.")
        return 0
    
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    all_documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(files):
        status_text.text(f"Processing: {file.name}")
        pdf_path = process_uploaded_file(file)
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_documents.extend(chunks)
        
        progress = (idx + 1) / len(files)
        progress_bar.progress(progress)

    if all_documents:
        status_text.text("Creating embeddings...")
        # Use HuggingFaceEmbeddings for compatibility
        embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_function,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        
        status_text.text(f"Successfully processed {len(all_documents)} document chunks")
        return len(all_documents)
    return 0

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODEL_NAME
    
    if "use_history" not in st.session_state:
        st.session_state.use_history = True
    
    if "app" not in st.session_state:
        st.session_state.app = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    
    if "active_service" not in st.session_state:
        st.session_state.active_service = "all"
    
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    
    if "model_details" not in st.session_state:
        st.session_state.model_details = {}
    
    if "last_models_refresh" not in st.session_state:
        st.session_state.last_models_refresh = datetime.now()
        
    # Database dimension settings
    if "target_dimension" not in st.session_state:
        st.session_state.target_dimension = 384  # Default embedding dimension
        
    if "vector_store_dir" not in st.session_state:
        st.session_state.vector_store_dir = PERSIST_DIRECTORY
        
    if "show_agents" not in st.session_state:
        st.session_state.show_agents = True

def refresh_available_models():
    """Refresh the list of available models from Ollama"""
    models = get_available_ollama_models()
    
    # Extract model names and details
    model_names = []
    model_details = {}
    
    for model in models:
        name = model.get("name")
        if name:
            model_names.append(name)
            # Store additional details
            model_details[name] = {
                "size": model.get("size", "Unknown"),
                "modified_at": model.get("modified_at", "Unknown"),
                "details": model.get("details", {})
            }
    
    st.session_state.available_models = model_names
    st.session_state.model_details = model_details
    st.session_state.last_models_refresh = datetime.now()
    
    return model_names

def model_selector_widget():
    """Widget for selecting and managing Ollama models"""
    st.subheader("Model Selection")
    
    # Refresh models button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Last refreshed: {st.session_state.last_models_refresh.strftime('%H:%M:%S')}")
    with col2:
        if st.button("🔄 Refresh"):
            with st.spinner("Refreshing models..."):
                refresh_available_models()
    
    # If no models are available yet, refresh
    if not st.session_state.available_models:
        with st.spinner("Loading available models..."):
            refresh_available_models()
    
    # Model selector
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Model",
            st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.model_name) 
                if st.session_state.model_name in st.session_state.available_models else 0
        )
        
        # Display model details if available
        if selected_model in st.session_state.model_details:
            details = st.session_state.model_details[selected_model]
            with st.expander("Model Details"):
                st.write(f"**Size:** {details.get('size', 'Unknown')}")
                st.write(f"**Modified:** {details.get('modified_at', 'Unknown')}")
                
                # Display additional details if available
                model_details = details.get("details", {})
                if model_details:
                    st.write("**Specifications:**")
                    for key, value in model_details.items():
                        st.write(f"- {key}: {value}")
        
        # Apply button
        if st.button("Apply Model"):
            if selected_model != st.session_state.model_name:
                with st.spinner(f"Switching to model: {selected_model}"):
                    # Update model name in session state
                    st.session_state.model_name = selected_model
                    # Reset app to apply new model
                    st.session_state.app = None
                    st.success(f"Switched to model: {selected_model}")
                    st.rerun()
    else:
        st.warning("No models available. Make sure Ollama is running.")
    
    # Pull new model section
    st.subheader("Pull New Model")
    
    # Common Ollama models
    common_models = [
        "llama3:8b", "llama3:70b", 
        "mistral:7b", "mistral:latest",
        "phi3:latest", "phi3:mini",
        "gemma:7b", "gemma:2b",
        "codellama:7b", "codellama:13b",
        "orca-mini:7b", "vicuna:7b"
    ]
    
    # Filter out models that are already available
    available_models_set = set(st.session_state.available_models)
    new_models = [model for model in common_models if model not in available_models_set]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Custom model input
        custom_model = st.text_input("Model Name", 
                                    placeholder="e.g., llama3:8b, mistral:7b, etc.")
    
    with col2:
        # Pull button
        pull_button = st.button("Pull Model")
    
    # Common models suggestions
    if new_models:
        with st.expander("Suggested Models"):
            st.write("Click to select:")
            for i in range(0, len(new_models), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(new_models):
                        model = new_models[i + j]
                        if cols[j].button(model, key=f"suggest_{model}"):
                            custom_model = model
                            st.session_state["custom_model"] = model
                            st.rerun()
    
    # Handle model pull
    if pull_button and custom_model:
        model_name = custom_model.strip()
        if model_name:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, percent):
                status_text.text(message)
                progress_bar.progress(percent / 100)
            
            with st.spinner(f"Pulling model: {model_name}"):
                success = pull_ollama_model(model_name, update_progress)
                
                if success:
                    st.success(f"Successfully pulled model: {model_name}")
                    # Refresh available models
                    refresh_available_models()
                    # Set as current model
                    st.session_state.model_name = model_name
                    # Reset app to use new model
                    st.session_state.app = None
                    st.rerun()
                else:
                    st.error(f"Failed to pull model: {model_name}")

def save_conversation_log(conversation_data):
    """Save conversation logs to JSON file"""
    log_dir = "conversation_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/conversation_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(conversation_data, f, indent=2)

def chat_interface():
    """Chat interface tab"""
    st.title("MultiAgent LLM Chat")
    
    # Agent display toggle
    show_agents = st.sidebar.checkbox("Show agent breakdown", value=st.session_state.show_agents)
    if show_agents != st.session_state.show_agents:
        st.session_state.show_agents = show_agents
        st.rerun()
    
    # Service selector
    available_services = ["all", "local_llm", "wikipedia", "chat"]
    if Config.OPENAI_API_KEY:
        available_services.append("openai")
    if Config.ANTHROPIC_API_KEY:
        available_services.append("anthropic")
    if Config.PERPLEXITY_API_KEY:
        available_services.append("perplexity")
    if Config.GEMINI_API_KEY:
        available_services.append("gemini")
    
    service_selector = st.sidebar.selectbox(
        "View responses from:",
        available_services,
        index=available_services.index(st.session_state.active_service)
    )
    
    if service_selector != st.session_state.active_service:
        st.session_state.active_service = service_selector
        st.rerun()
        
    # Agent roles explanation
    with st.sidebar.expander("About Multi-Agent Responses"):
        st.markdown("""
        Responses are generated by multiple specialized agents:
        
        - **Researcher**: Provides detailed analysis with citations
        - **Summarizer**: Gives concise summary of key points
        - **Critic**: Identifies potential weaknesses or biases
        
        Toggle 'Show agent breakdown' to see individual agent responses.
        """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display responses if this is a user message
            if message["role"] == "user" and message["id"] in st.session_state.responses:
                responses = st.session_state.responses[message["id"]]
                
                # Display responses based on selected service
                if st.session_state.active_service == "all":
                    # Display all responses in tabs
                    tabs = st.tabs([s.title() for s in responses.keys()])
                    for i, (service, response) in enumerate(responses.items()):
                        with tabs[i]:
                            if response.get("error", False):
                                st.error(response["content"])
                            else:
                                st.markdown(response["content"])
                                
                                # Show sources for local_llm
                                if service == "local_llm" and "sources" in response:
                                    with st.expander("View Source Documents"):
                                        for i, document in enumerate(response["sources"]):
                                            st.markdown(f"**Source {i+1}:** {document.metadata.get('source', 'Unknown')}")
                                            st.markdown(document.page_content)
                                            st.markdown("---")
                else:
                    # Display only the selected service
                    service = st.session_state.active_service
                    if service in responses:
                        response = responses[service]
                        if response.get("error", False):
                            st.error(response["content"])
                        else:
                            st.markdown(response["content"])
                            
                            # Show sources for local_llm
                            if service == "local_llm" and "sources" in response:
                                with st.expander("View Source Documents"):
                                    for i, document in enumerate(response["sources"]):
                                        st.markdown(f"**Source {i+1}:** {document.metadata.get('source', 'Unknown')}")
                                        st.markdown(document.page_content)
                                        st.markdown("---")
                    else:
                        st.info(f"No response available from {service.title()}")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        message_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "id": message_id})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and get responses
        with st.chat_message("assistant"):
            with st.spinner("Processing your query across multiple services..."):
                try:
                    # Show a message that we're initializing the LLM
                    st.info("Initializing MultiAgent LLM system...")
                    
                    # Import the MultiAgentLLM class from run_localGPTMultiAgent2.py
                    import sys
                    import importlib.util
                    
                    # Check if required modules are available
                    required_modules = ['anthropic', 'openai', 'google.generativeai', 'bs4', 'googlesearch', 
                                       'langchain_community', 'langchain_core', 'transformers', 'sentence_transformers']
                    
                    missing_modules = []
                    for module in required_modules:
                        if importlib.util.find_spec(module) is None:
                            missing_modules.append(module)
                    
                    if missing_modules:
                        st.error(f"Missing required modules: {', '.join(missing_modules)}")
                        st.info("Please install the missing modules with pip")
                        raise ImportError(f"Missing required modules: {', '.join(missing_modules)}")
                    
                    # Use subprocess method directly - more reliable
                    st.info("Using subprocess method to run the MultiAgent LLM...")
                    
                    # Set up the command with appropriate flags
                    history_flag = "--use_history" if st.session_state.use_history else ""
                    save_flag = "--save_responses"  # Always save responses for debugging
                    command = f"python3 run_localGPTMultiAgent2.py {history_flag} {save_flag}"
                    st.info(f"Running command: {command}")
                    
                    # Run the command and capture the output
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Send the query to the process
                    st.info(f"Sending query: {prompt}")
                    stdout, stderr = process.communicate(input=prompt)
                    
                    if stderr and "Error" in stderr:
                        st.error(f"Error from subprocess: {stderr}")
                    
                    # Parse the responses
                    parsed_responses = {}
                    current_service = None
                    current_content = []
                    sources = []
                    in_sources = False
                    
                    # Log the raw output for debugging
                    st.info(f"Got response from subprocess. Parsing {len(stdout)} characters of output.")
                    
                    for line in stdout.split("\n"):
                        if line.startswith("> "):
                            # If we were processing a previous service, save it
                            if current_service:
                                if current_service == "local_llm" and sources:
                                    parsed_responses[current_service] = {
                                        "content": "\n".join(current_content),
                                        "sources": sources
                                    }
                                else:
                                    parsed_responses[current_service] = {"content": "\n".join(current_content)}
                                current_content = []
                                sources = []
                            
                            # Extract service name and start of response
                            parts = line[2:].split(" Response: ", 1)
                            if len(parts) == 2:
                                current_service = parts[0].lower()
                                current_content.append(parts[1])
                        elif line.startswith("--- Source Documents ---"):
                            # Start of source documents section
                            in_sources = True
                        elif line.startswith("--- End of Sources ---"):
                            # End of source documents section
                            in_sources = False
                        elif current_service and line.startswith("Source: "):
                            # This is a source document
                            source_info = line[8:]  # Remove "Source: " prefix
                            sources.append({"metadata": {"source": source_info}, "page_content": ""})
                        elif current_service and sources and line:
                            # This is content for the current source
                            if sources[-1]["page_content"]:
                                sources[-1]["page_content"] += "\n" + line
                            else:
                                sources[-1]["page_content"] = line
                        elif current_service and line:
                            current_content.append(line)
                    
                    # Save the last service if there is one
                    if current_service:
                        if current_service == "local_llm" and sources:
                            parsed_responses[current_service] = {
                                "content": "\n".join(current_content),
                                "sources": sources
                            }
                        else:
                            parsed_responses[current_service] = {"content": "\n".join(current_content)}
                    
                    # If no responses were parsed, use the raw output
                    if not parsed_responses:
                        parsed_responses["output"] = {"content": stdout}
                        st.warning("No structured responses found in the output. Showing raw output.")
                
                    # Store responses in session state
                    st.session_state.responses[message_id] = parsed_responses
                    
                    # Display responses based on selected service
                    if st.session_state.active_service == "all":
                        # Display all responses in tabs
                        tabs = st.tabs([s.title() for s in parsed_responses.keys()])
                        for i, (service, response) in enumerate(parsed_responses.items()):
                            with tabs[i]:
                                st.markdown(response["content"])
                    else:
                        # Display only the selected service
                        service = st.session_state.active_service
                        if service in parsed_responses:
                            response = parsed_responses[service]
                            st.markdown(response["content"])
                        else:
                            st.info(f"No response available from {service.title()}")
                    
                    # Add assistant message to chat history (use local_llm response as default)
                    if "local_llm" in parsed_responses:
                        assistant_response = parsed_responses["local_llm"]["content"]
                    else:
                        # Find the first response
                        assistant_response = "I couldn't generate a response at the moment."
                        for service, response in parsed_responses.items():
                            assistant_response = response["content"]
                            break
                    
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "id": f"response_{message_id}"})
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "id": f"response_{message_id}"})

def ingestion_interface():
    """Document ingestion tab"""
    st.title("PDF Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process PDFs") and uploaded_files:
        with st.spinner("Processing documents..."):
            chunks_processed = ingest_pdfs(uploaded_files)
            if chunks_processed:
                st.success(f"Successfully processed {chunks_processed} chunks")
                st.info(f"Database location: {PERSIST_DIRECTORY}")
                
                # Reinitialize app to use the new database
                st.session_state.app = None
                st.rerun()
    
    if st.button("View Database Info"):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Use HuggingFaceEmbeddings for compatibility
            embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
            
            db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_function
            )
            st.write(f"Total documents in database: {db._collection.count()}")
        except Exception as e:
            st.error(f"Error accessing database: {e}")

def manage_chroma_dimensions(collection, target_dim=384):
    """Ensure Chroma collection has correct dimensions"""
    if collection.count() > 0:
        sample = collection.peek(1)['embeddings'][0]
        current_dim = len(sample)
        if current_dim != target_dim:
            st.warning(f"Adjusting Chroma dimensions from {current_dim} to {target_dim}")
            
            # Get all documents and embeddings
            docs = collection.get(include=["documents", "embeddings"])
            texts = docs["documents"]
            old_embeddings = docs["embeddings"]
            
            # Adjust dimensions
            new_embeddings = []
            for emb in old_embeddings:
                if current_dim > target_dim:
                    # Truncate
                    new_embeddings.append(emb[:target_dim])
                else:
                    # Pad with zeros
                    new_embeddings.append(emb + [0.0] * (target_dim - current_dim))
            
            # Delete old collection
            collection.delete(ids=docs["ids"])
            
            # Create new collection with adjusted dimensions
            collection.add(
                embeddings=new_embeddings,
                documents=texts,
                ids=docs["ids"],
                metadatas=docs["metadatas"]
            )
            
            return True
    return False

def database_management_interface():
    """Database management tab"""
    st.title("Vector Database Management")
    
    # Dimension management
    st.subheader("Embedding Dimensions")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        target_dimension = st.number_input(
            "Target Embedding Dimension",
            min_value=128,
            max_value=8192,
            value=st.session_state.target_dimension,
            step=128,
            help="Set the dimension for embeddings in your vector database"
        )
    
    with col2:
        if st.button("Apply Dimension"):
            if target_dimension != st.session_state.target_dimension:
                st.session_state.target_dimension = target_dimension
                st.success(f"Target dimension set to {target_dimension}")
                st.rerun()
    
    # Database information
    with st.expander("Database Information"):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Use HuggingFaceEmbeddings for compatibility
            embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
            
            db = Chroma(
                persist_directory=st.session_state.vector_store_dir,
                embedding_function=embedding_function
            )
            
            count = db._collection.count()
            st.write(f"Total documents: {count}")
            
            if count > 0:
                collection_info = db._collection.get()
                if collection_info and 'embeddings' in collection_info and len(collection_info['embeddings']) > 0:
                    dimension = len(collection_info['embeddings'][0])
                    st.info(f"Current embedding dimension: {dimension}")
                    
                    if dimension != st.session_state.target_dimension:
                        st.warning(f"⚠️ Dimension mismatch! Collection: {dimension}, Target: {st.session_state.target_dimension}")
                        if st.button("Fix Dimension Mismatch"):
                            with st.spinner("Adjusting dimensions..."):
                                if manage_chroma_dimensions(db._collection, st.session_state.target_dimension):
                                    st.success("Dimensions adjusted successfully!")
                                    st.rerun()
                    else:
                        st.success("✅ Dimensions match!")
                    
                    # Sample a document to show content
                    if 'documents' in collection_info:
                        with st.expander("Sample document"):
                            st.write(collection_info['documents'][0][:500] + "...")
                else:
                    st.warning("No embeddings found in the database.")
            else:
                st.info("Database is empty - no documents to display")
        except Exception as e:
            st.error(f"Error accessing database: {e}")
    
    st.subheader("Reset Database")
    if st.button("Reset Database", type="primary", help="Deletes the current vector database"):
        try:
            import shutil
            if os.path.exists(st.session_state.vector_store_dir):
                shutil.rmtree(st.session_state.vector_store_dir)
                st.success("Database deleted successfully. Upload new documents to create a fresh database.")
                
                # Reinitialize app to use the new database
                st.session_state.app = None
                st.rerun()
            else:
                st.info("No database found to delete.")
        except Exception as e:
            st.error(f"Error deleting database: {e}")

def settings_interface():
    """Settings tab"""
    st.title("Settings")
    
    # History toggle
    use_history = st.checkbox("Use conversation history", value=st.session_state.use_history)
    
    # Apply settings button
    if st.button("Apply Settings"):
        # Check if settings changed
        if use_history != st.session_state.use_history:
            st.session_state.use_history = use_history
            
            # Reset app to apply new settings
            st.session_state.app = None
            st.success("Settings applied! Reinitializing app...")
            st.rerun()
    
    # API Keys
    st.subheader("API Keys")
    
    # Display current API key status
    missing_keys = Config.validate()
    
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}")
    else:
        st.success("All API keys are configured")
    
    # OpenAI API Key
    openai_key = st.text_input("OpenAI API Key", value=Config.OPENAI_API_KEY or "", type="password")
    
    # Anthropic API Key
    anthropic_key = st.text_input("Anthropic API Key", value=Config.ANTHROPIC_API_KEY or "", type="password")
    
    # Perplexity API Key
    perplexity_key = st.text_input("Perplexity API Key", value=Config.PERPLEXITY_API_KEY or "", type="password")
    
    # Gemini API Key
    gemini_key = st.text_input("Gemini API Key", value=Config.GEMINI_API_KEY or "", type="password")
    
    # Save API keys button
    if st.button("Save API Keys"):
        # Create .env file if it doesn't exist
        env_path = ".env"
        env_content = ""
        
        # Read existing .env file if it exists
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                env_content = f.read()
        
        # Update or add OpenAI API key
        if "OPENAI_API_KEY" in env_content:
            env_content = re.sub(r'OPENAI_API_KEY=.*', f'OPENAI_API_KEY={openai_key}', env_content)
        else:
            env_content += f'\nOPENAI_API_KEY={openai_key}'
        
        # Update or add Anthropic API key
        if "ANTHROPIC_API_KEY" in env_content:
            env_content = re.sub(r'ANTHROPIC_API_KEY=.*', f'ANTHROPIC_API_KEY={anthropic_key}', env_content)
        else:
            env_content += f'\nANTHROPIC_API_KEY={anthropic_key}'
        
        # Update or add Perplexity API key
        if "PERPLEXITY_API_KEY" in env_content:
            env_content = re.sub(r'PERPLEXITY_API_KEY=.*', f'PERPLEXITY_API_KEY={perplexity_key}', env_content)
        else:
            env_content += f'\nPERPLEXITY_API_KEY={perplexity_key}'
        
        # Update or add Gemini API key
        if "GEMINI_API_KEY" in env_content:
            env_content = re.sub(r'GEMINI_API_KEY=.*', f'GEMINI_API_KEY={gemini_key}', env_content)
        else:
            env_content += f'\nGEMINI_API_KEY={gemini_key}'
        
        # Write updated .env file
        with open(env_path, "w") as f:
            f.write(env_content.strip())
        
        st.success("API keys saved! Please restart the application for changes to take effect.")
    
    # Reset button
    if st.button("Reset All Settings"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("All settings reset! Reinitializing app...")
        st.rerun()

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="MultiAgent LLM System", layout="wide")
    initialize_session_state()
    
    # Add sidebar with connection information and model selector
    with st.sidebar:
        st.title("MultiAgent LLM System")
        st.caption("Version 2.0")
        
        # Model selector widget
        model_selector_widget()
        
        # Empty placeholder for spacing
        st.empty()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Document Ingestion", "Database Management", "Settings", "System Info"])
    
    with tab1:
        chat_interface()
        
    with tab2:
        ingestion_interface()
        
    with tab3:
        database_management_interface()
        
    with tab4:
        settings_interface()
        
    with tab5:
        st.title("System Information")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            st.success("CUDA is available!")
            st.write(f"PyTorch version: {torch.__version__}")
            st.write(f"CUDA version: {torch.version.cuda}")
            st.write(f"Number of CUDA devices: {torch.cuda.device_count()}")
            st.write(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            st.error("CUDA is not available. Performance may be slower.")
            st.write(f"PyTorch version: {torch.__version__}")
        
        # Display Python information
        st.subheader("Python Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"Python executable: {sys.executable}")
        
        # Display OS information
        st.subheader("Operating System Information")
        import platform
        st.write(f"OS: {platform.system()} {platform.release()}")
        st.write(f"Platform: {platform.platform()}")
        
        # Display current working directory
        st.subheader("Working Directory")
        st.write(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
