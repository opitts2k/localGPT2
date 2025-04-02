import os
import asyncio
import streamlit as st
import logging
import sys
import subprocess
import json
import requests
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
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

# Shared Constants
OLLAMA_HOST = "http://localhost:11434"  # Ollama server base URL
INFERENCE_SERVER_URL = f"{OLLAMA_HOST}/api/chat"  # Ollama chat API endpoint
EMBEDDING_SERVER_URL = f"{OLLAMA_HOST}/api/embeddings"  # Ollama embeddings endpoint
DEFAULT_MODEL_NAME = "llama3:8b"  # Default model
PDF_DIRECTORY = "./pdfs"
PERSIST_DIRECTORY = "./DB/SLAVE"  # Use local DB directory
RESPONSES_DIRECTORY = "./llmresponses"  # Directory to save LLM responses

# Import torch separately for system information
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    logger.error(f"Error importing torch: {e}")
    st.error(f"Error importing torch: {e}")

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    
    if "active_service" not in st.session_state:
        st.session_state.active_service = "all"
    
    if "use_history" not in st.session_state:
        st.session_state.use_history = True
        
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL_NAME
        
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
        
    if "last_models_refresh" not in st.session_state:
        st.session_state.last_models_refresh = datetime.now()

def get_available_ollama_models():
    """
    Get a list of available Ollama models.
    Returns a list of model names.
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models if model.get("name")]
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
        if model_name not in models:
            logger.info(f"Model {model_name} not found. Pulling from Ollama repository...")
            return pull_ollama_model(model_name, progress_callback)
        else:
            logger.info(f"Model {model_name} is already available")
            if progress_callback:
                progress_callback(f"Model {model_name} is already available", 100)
            return True
    except Exception as e:
        logger.error(f"Error ensuring Ollama model: {e}")
        logger.error("Make sure Ollama server is running at: " + OLLAMA_HOST)
        if progress_callback:
            progress_callback(f"Error: {str(e)}", 0)
        return False

def refresh_available_models():
    """Refresh the list of available models from Ollama"""
    models = get_available_ollama_models()
    st.session_state.available_models = models
    st.session_state.last_models_refresh = datetime.now()
    return models

def run_query_with_ollama(query, model_name, use_history=False):
    """
    Run a query using the Ollama API directly.
    
    Args:
        query: The query to run
        model_name: The model to use
        use_history: Whether to use conversation history
        
    Returns:
        dict: The parsed responses from different services
    """
    try:
        # Ensure the model is available
        ensure_ollama_model(model_name)
        
        # Format the query for the Ollama API
        headers = {"Content-Type": "application/json"}
        
        # Format messages for Ollama API
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": query}
        ]
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }
        
        # Send the request to Ollama
        response = requests.post(INFERENCE_SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        
        if "message" in response_json and "content" in response_json["message"]:
            content = response_json["message"]["content"]
            
            # Create a simple response dictionary
            return {
                "local_llm": {"content": content},
                "ollama": {"content": content}
            }
        else:
            logger.warning("Unexpected response format from LLM server.")
            return {
                "error": {"content": "Unexpected response format from LLM server."}
            }
    except Exception as e:
        logger.error(f"Error running query with Ollama: {e}")
        return {
            "error": {"content": f"Error running query with Ollama: {str(e)}"}
        }

def run_query_with_script(query, model_name, use_history=False):
    """
    Run a query using the run_localGPTMultiAgent4.py script.
    
    Args:
        query: The query to run
        model_name: The model to use
        use_history: Whether to use conversation history
        
    Returns:
        dict: The parsed responses from different services
    """
    try:
        # Build the command
        history_flag = "--use_history" if use_history else ""
        command = f"python run_localGPTMultiAgent4.py {history_flag} --model {model_name}"
        
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
        stdout, stderr = process.communicate(input=query)
        
        if stderr:
            logger.error(f"Error from script: {stderr}")
        
        # Parse the responses
        parsed_responses = {}
        current_service = None
        current_content = []
        
        for line in stdout.split("\n"):
            if line.startswith("> "):
                # If we were processing a previous service, save it
                if current_service:
                    parsed_responses[current_service] = {"content": "\n".join(current_content)}
                    current_content = []
                
                # Extract service name and start of response
                parts = line[2:].split(" Response: ", 1)
                if len(parts) == 2:
                    current_service = parts[0].lower()
                    current_content.append(parts[1])
            elif current_service and line:
                current_content.append(line)
        
        # Save the last service if there is one
        if current_service and current_content:
            parsed_responses[current_service] = {"content": "\n".join(current_content)}
        
        # If no responses were parsed, use the raw output
        if not parsed_responses:
            parsed_responses["output"] = {"content": stdout}
        
        return parsed_responses
    except Exception as e:
        logger.error(f"Error running query with script: {e}")
        return {
            "error": {"content": f"Error running query with script: {str(e)}"}
        }

def save_response_to_file(query, responses):
    """Save responses to a file in the responses directory"""
    try:
        # Create responses directory if it doesn't exist
        if not os.path.exists(RESPONSES_DIRECTORY):
            os.makedirs(RESPONSES_DIRECTORY)
            logger.info(f"Created responses directory at {RESPONSES_DIRECTORY}")
        
        # Create a safe filename from the query
        safe_query = re.sub(r'[^\w\s-]', '', query.lower())
        safe_query = re.sub(r'[\s-]+', '_', safe_query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_query}_{timestamp}.txt"
        filepath = os.path.join(RESPONSES_DIRECTORY, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for service, response in responses.items():
                f.write(f"=== {service.upper()} RESPONSE ===\n")
                if isinstance(response, Exception):
                    f.write(f"Error: {str(response)}\n")
                else:
                    f.write(f"{response['content']}\n")
                    if service == "local_llm" and response.get("sources"):
                        f.write("\n--- Source Documents ---\n")
                        for source in response["sources"]:
                            f.write(f"\nSource: {source.metadata.get('source', 'Unknown')}\n")
                            f.write(f"{source.page_content}\n")
                        f.write("--- End of Sources ---\n")
                f.write("\n\n")
            
        logger.info(f"Saved response to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving response to file: {e}")
        return None

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
        
        # Apply button
        if st.button("Apply Model"):
            if selected_model != st.session_state.model_name:
                with st.spinner(f"Switching to model: {selected_model}"):
                    # Update model name in session state
                    st.session_state.model_name = selected_model
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
                            st.session_state["custom_model_input"] = model
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
                    st.rerun()
                else:
                    st.error(f"Failed to pull model: {model_name}")

def main():
    st.set_page_config(page_title="MultiAgent LLM System", layout="wide")
    initialize_session_state()
    
    st.title("MultiAgent LLM System")
    st.caption("A system that uses multiple AI services to answer your questions")
    
    # Sidebar for settings and model selection
    with st.sidebar:
        st.header("Settings")
        
        # Use history toggle
        use_history = st.checkbox("Use conversation history", value=st.session_state.use_history)
        
        # Service selector
        available_services = ["all", "local_llm", "wikipedia", "chat", "openai", "anthropic", "perplexity", "gemini"]
        service_selector = st.selectbox(
            "View responses from:",
            available_services,
            index=available_services.index(st.session_state.active_service)
        )
        
        # Apply settings button
        if st.button("Apply Settings"):
            # Check if settings changed
            if use_history != st.session_state.use_history or service_selector != st.session_state.active_service:
                st.session_state.use_history = use_history
                st.session_state.active_service = service_selector
                st.success("Settings applied!")
                st.rerun()
        
        # Add model selector widget
        model_selector_widget()
    
    # Display system information
    with st.sidebar:
        st.header("System Information")
        
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
        
        # Display Ollama information
        st.subheader("Ollama Information")
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                version = response.json().get("version", "Unknown")
                st.success(f"Ollama is running (version: {version})")
            else:
                st.error("Ollama is not responding correctly")
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            st.warning("Make sure Ollama is running at: " + OLLAMA_HOST)
    
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
                            if isinstance(response, Exception):
                                st.error(f"Error: {str(response)}")
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
                        if isinstance(response, Exception):
                            st.error(f"Error: {str(response)}")
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
            with st.spinner(f"Processing your query using {st.session_state.model_name}..."):
                try:
                    # Use direct Ollama API for faster response
                    parsed_responses = run_query_with_ollama(
                        prompt, 
                        st.session_state.model_name, 
                        st.session_state.use_history
                    )
                    
                    # Save responses to file
                    save_response_to_file(prompt, parsed_responses)
                    
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
                    elif "ollama" in parsed_responses:
                        assistant_response = parsed_responses["ollama"]["content"]
                    else:
                        # Find the first response
                        assistant_response = "I couldn't generate a response at the moment."
                        for service, response in parsed_responses.items():
                            if service != "error":
                                assistant_response = response["content"]
                                break
                    
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "id": f"response_{message_id}"})
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "id": f"response_{message_id}"})

if __name__ == "__main__":
    main()
