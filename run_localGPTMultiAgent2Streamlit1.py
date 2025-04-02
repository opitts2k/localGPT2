import os
import asyncio
import streamlit as st
import logging
import sys
from typing import Dict, Any
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

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "app" not in st.session_state:
        st.session_state.app = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    
    if "active_service" not in st.session_state:
        st.session_state.active_service = "all"
    
    if "use_history" not in st.session_state:
        st.session_state.use_history = True

def main():
    st.set_page_config(page_title="MultiAgent LLM System", layout="wide")
    initialize_session_state()
    
    st.title("MultiAgent LLM System")
    st.caption("A system that uses multiple AI services to answer your questions")
    
    # Sidebar for settings
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
                
                # Reset app to apply new settings
                st.session_state.app = None
                st.success("Settings applied! Reinitializing app...")
                st.rerun()
    
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
        import datetime
        message_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "id": message_id})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and get responses
        with st.chat_message("assistant"):
            with st.spinner("Processing your query across multiple services..."):
                try:
                    # Run the command line version of the LLM
                    history_flag = "--use_history" if st.session_state.use_history else ""
                    command = f"python run_localGPTMultiAgent2.py {history_flag}"
                    
                    # Show a message that we're running the command
                    st.info(f"Running command: {command}")
                    
                    # Create a temporary file to store the query
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                        f.write(prompt)
                        query_file = f.name
                    
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
                    stdout, stderr = process.communicate(input=prompt)
                    
                    if stderr:
                        st.error(f"Error: {stderr}")
                    
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

if __name__ == "__main__":
    main()
