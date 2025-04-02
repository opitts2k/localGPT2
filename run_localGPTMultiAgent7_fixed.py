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

# Import our minimal Anthropic interface
from anthropic_minimal import MinimalAnthropic
import openai
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search
from testcommandprompt2 import OpenSourceApp
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from prompt_template_utils import get_prompt_template
from constants import (
    MODELS_PATH,
    MAX_NEW_TOKENS,
    CHROMA_SETTINGS
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Shared Constants
OLLAMA_HOST = "http://localhost:11434"  # Ollama server base URL
INFERENCE_SERVER_URL = f"{OLLAMA_HOST}/api/chat"  # Ollama chat API endpoint
EMBEDDING_SERVER_URL = f"{OLLAMA_HOST}/api/embeddings"  # Ollama embeddings endpoint
MODEL_NAME = "llama3:8b"  # Default model
EMBEDDING_MODEL_NAME = None  # Will use the same as MODEL_NAME by default
PDF_DIRECTORY = "./pdfs"
# Default vector store directory - will be updated from session state if available
PERSIST_DIRECTORY = "./DB/SLAVE"  # Use local DB directory
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

def handle_service_error(func):
    """Decorator for handling service errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise ServiceError(
                service=func.__name__,
                message="Service temporarily unavailable",
                original_error=e
            )
    return wrapper

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

class OllamaEmbeddings(Embeddings):
    """Embeddings implementation using Ollama's embedding API with dimension handling."""
    
    def __init__(self, model_name=None, target_dimension=None):
        self.model_name = model_name or MODEL_NAME
        self.embedding_url = EMBEDDING_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.target_dimension = target_dimension  # Target dimension for compatibility
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to the embedding server."""
        try:
            # Simple test with a short text
            self.embed_query("test")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama embedding server: {e}")
            return False
    
    def embed_documents(self, texts):
        """Embed multiple documents."""
        return [self.embed_query(text) for text in texts]

    def _adjust_dimension(self, embedding, target_dim):
        """Adjust embedding dimension to match target dimension."""
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate to target dimension
            return embedding[:target_dim]
        else:
            # Pad with zeros to reach target dimension
            return embedding + [0.0] * (target_dim - current_dim)

    def embed_query(self, text):
        """Embed a single text query."""
        # Ollama API format for embeddings
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            # Send request to Ollama API
            response = requests.post(self.embedding_url, json=payload, headers=self.headers)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract embedding from response
            if "embedding" in response_json:
                embedding = response_json["embedding"]
                if isinstance(embedding, list) and len(embedding) > 0:
                    # Adjust dimension if target dimension is specified
                    if self.target_dimension is not None:
                        return self._adjust_dimension(embedding, self.target_dimension)
                    return embedding
                else:
                    raise ValueError(f"Invalid embedding format: {type(embedding)}")
            else:
                raise ValueError(f"Unexpected response format. Keys: {list(response_json.keys())}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with the embedding server: {e}")
            # Return a zero vector as fallback (better than None for some operations)
            dim = self.target_dimension if self.target_dimension is not None else 4096
            return [0.0] * dim
        except ValueError as e:
            logger.error(str(e))
            dim = self.target_dimension if self.target_dimension is not None else 4096
            return [0.0] * dim

class OllamaLLM(LLM):
    """LLM implementation for Ollama API."""
    
    model_name: str = MODEL_NAME
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Ollama API and return the response."""
        headers = {"Content-Type": "application/json"}
        
        # Format messages for Ollama API
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        
        try:
            response = requests.post(INFERENCE_SERVER_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            if "message" in response_json and "content" in response_json["message"]:
                return response_json["message"]["content"]
            else:
                logger.warning("Unexpected response format from LLM server.")
                return "I'm sorry, I couldn't generate a response at the moment."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with the inference server: {e}")
            return "An error occurred while communicating with the LLM server."
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}

class APIClientManager:
    """Manages API clients for different services"""
    def __init__(self, device_type: str = "cuda"):
        Config.validate()
        self.openai_client = self._init_openai()
        self.anthropic_client = self._init_anthropic()
        self.perplexity_client = self._init_perplexity()
        self.gemini_client = self._init_gemini()
        self.wiki = WikipediaAPIWrapper()
        
        # Try to initialize with CUDA, fall back to CPU if it fails
        try:
            logger.info(f"Initializing OpenSourceApp with device: {device_type}")
            self.zuck_bot = OpenSourceApp(device=device_type)
        except RuntimeError as e:
            if "Could not find any implementations for backend: cuda" in str(e) or "libcudart.so" in str(e):
                logger.warning(f"Failed to initialize with CUDA: {e}")
                logger.warning("Falling back to CPU for OpenSourceApp")
                self.zuck_bot = OpenSourceApp(device="cpu")
            else:
                raise
        
    def _init_openai(self):
        try:
            if Config.OPENAI_API_KEY:
                return openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            return None
    
    def _init_anthropic(self):
        try:
            if Config.ANTHROPIC_API_KEY:
                return MinimalAnthropic(api_key=Config.ANTHROPIC_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing Anthropic client: {e}")
            return None
    
    def _init_perplexity(self):
        try:
            if Config.PERPLEXITY_API_KEY:
                return ChatPerplexity(api_key=Config.PERPLEXITY_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing Perplexity client: {e}")
            return None
    
    def _init_gemini(self):
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                return genai.GenerativeModel('gemini-pro')
            return None
        except Exception as e:
            logger.warning(f"Error initializing Gemini client: {e}")
            return None

class LLMManager:
    """Manages local LLM operations"""
    def __init__(self, model_name: str, use_history: bool):
        self.model_name = model_name
        self.use_history = use_history
        self.embedding_model_name = EMBEDDING_MODEL_NAME or model_name
        
        logger.info(f"Initializing LLM Manager with model: {model_name}, history: {use_history}")
        
        # Ensure the model is available in Ollama
        ensure_ollama_model(model_name)
        
        # Initialize QA pipeline
        self.qa = self._init_qa_pipeline()
        
    def _init_qa_pipeline(self):
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        
        # Use OllamaEmbeddings
        embeddings = OllamaEmbeddings(model_name=self.embedding_model_name, target_dimension=768)
        
        # Use the vector store directory from session state
        persist_dir = st.session_state.vector_store_dir
        logger.info(f"Connecting to vector database at: {persist_dir}")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            logger.warning(f"Created empty vector database directory at {persist_dir}")
            logger.warning("You may need to ingest documents before querying")
        
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        retriever = db.as_retriever()
        
        logger.info("Setting up prompt template")
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=self.use_history)
        
        logger.info(f"Loading LLM model: {self.model_name}")
        
        # Use OllamaLLM
        llm = OllamaLLM(model_name=self.model_name, temperature=0.7)
        
        qa_kwargs = {
            "llm": llm,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "chain_type_kwargs": {"prompt": prompt}
        }
        
        if self.use_history:
            qa_kwargs["chain_type_kwargs"]["memory"] = memory
        
        logger.info("Initializing QA pipeline")
        return RetrievalQA.from_chain_type(**qa_kwargs)

class ResponseManager:
    """Manages responses from different services"""
    def __init__(self, api_clients: APIClientManager, llm_manager: LLMManager):
        self.clients = api_clients
        self.llm = llm_manager
        
        # Determine which services are available
        available_services = ["local_llm", "wikipedia"]
        if api_clients.openai_client:
            available_services.append("openai")
        if api_clients.anthropic_client:
            available_services.append("anthropic")
        if api_clients.perplexity_client:
            available_services.append("perplexity")
        if api_clients.gemini_client:
            available_services.append("gemini")
            
        logger.info("Response Manager initialized with services: " + ", ".join(available_services))
        
    async def get_responses(self, query: str) -> Dict[str, Any]:
        """Get responses from all services concurrently"""
        logger.info(f"Processing query: {query}")
        
        # Create tasks for all services
        tasks = [
            self._get_local_llm(query),
            self._get_wiki_research(query),
            self._get_chat_output(query)
        ]
        
        # Add optional API services if available
        if self.clients.openai_client:
            tasks.append(self._get_openai_response(query))
        if self.clients.anthropic_client:
            tasks.append(self._get_anthropic_response(query))
        if self.clients.perplexity_client:
            tasks.append(self._get_perplexity_response(query))
        if self.clients.gemini_client:
            tasks.append(self._get_gemini_response(query))
            
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build response dictionary
        response_dict = {
            "local_llm": results[0],
            "wikipedia": results[1],
            "chat": results[2]
        }
        
        # Add optional API responses
        result_index = 3
        if self.clients.openai_client:
            response_dict["openai"] = results[result_index]
            result_index += 1
        if self.clients.anthropic_client:
            response_dict["anthropic"] = results[result_index]
            result_index += 1
        if self.clients.perplexity_client:
            response_dict["perplexity"] = results[result_index]
            result_index += 1
        if self.clients.gemini_client:
            response_dict["gemini"] = results[result_index]
        
        return response_dict
    
    @handle_service_error
    async def _get_local_llm(self, query: str) -> Dict:
        logger.info("Querying local LLM")
        res = self.llm.qa(query)
        return {
            "content": res["result"],
            "sources": res.get("source_documents", [])
        }
    
    @handle_service_error
    async def _get_wiki_research(self, query: str) -> Dict:
        logger.info("Querying Wikipedia")
        return {"content": self.clients.wiki.run(query)}
    
    @handle_service_error
    async def _get_chat_output(self, query: str) -> Dict:
        logger.info("Querying web search")
        try:
            # Use only the parameters that are supported by the search function
            urls = []
            try:
                # Try with no parameters first
                urls = [i for i in search(query)][:5]  # Get first 5 results
            except Exception as e:
                logger.warning(f"Basic search failed: {e}")
                # If that fails, try with just the query
                try:
                    urls = [i for i in search(query)][:5]
                except Exception as e:
                    logger.error(f"All search attempts failed: {e}")
            
            for url in urls:
                try:
                    self.clients.zuck_bot.add("web_page", url)
                except Exception as e:
                    logger.warning(f"Failed to add URL {url}: {e}")
                    continue
            return {"content": self.clients.zuck_bot.query(query)}
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {"content": "Web search unavailable at the moment."}
    
    @handle_service_error
    async def _get_openai_response(self, query: str) -> Dict:
        if self.clients.openai_client is None:
            raise ServiceError(
                service="_get_openai_response",
                message="OpenAI client not available",
                original_error=None
            )
        
        logger.info("Querying OpenAI")
        response = self.clients.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}],
            max_tokens=1000
        )
        return {"content": response.choices[0].message.content}
    
    @handle_service_error
    async def _get_anthropic_response(self, query: str) -> Dict:
        if self.clients.anthropic_client is None:
            raise ServiceError(
                service="_get_anthropic_response",
                message="Anthropic client not available",
                original_error=None
            )
        
        logger.info("Querying Anthropic")
        response = await self.clients.anthropic_client.generate_response(
            prompt=query,
            max_tokens=1000,
            model="claude-3-5-sonnet-latest"
        )
        return {"content": response}
    
    @handle_service_error
    async def _get_perplexity_response(self, query: str) -> Dict:
        if self.clients.perplexity_client is None:
            raise ServiceError(
                service="_get_perplexity_response",
                message="Perplexity client not available",
                original_error=None
            )
        
        logger.info("Querying Perplexity")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{query}")
        ])
        chain = prompt_template | self.clients.perplexity_client
        response = chain.invoke({"query": query})
        return {"content": response.content}
    
    @handle_service_error
    async def _get_gemini_response(self, query: str) -> Dict:
        if self.clients.gemini_client is None:
            raise ServiceError(
                service="_get_gemini_response",
                message="Gemini client not available",
                original_error=None
            )
        
        logger.info("Querying Gemini")
        response = self.clients.gemini_client.generate_content(query)
        return {"content": response.text}

class OutputFormatter:
    """Formats the output for display and saves responses to files"""
    def __init__(self):
        # Create responses directory if it doesn't exist
        if not os.path.exists(RESPONSES_DIRECTORY):
            os.makedirs(RESPONSES_DIRECTORY)
            logger.info(f"Created responses directory at {RESPONSES_DIRECTORY}")
    
    def format_response(self, response_data: Dict, query: str) -> Dict[str, Any]:
        """Format the response for display and save to file"""
        formatted_responses = {}
        
        for service, response in response_data.items():
            if isinstance(response, ServiceError):
                formatted_responses[service] = {
                    "content": f"Error: {response.message}",
                    "error": True
                }
            elif isinstance(response, Exception):
                formatted_responses[service] = {
                    "content": f"Error: {str(response)}",
                    "error": True
                }
            else:
                formatted_responses[service] = {
                    "content": response["content"],
                    "error": False
                }
                if service == "local_llm" and response.get("sources"):
                    formatted_responses[service]["sources"] = response["sources"]
        
        # Save responses to file
        self._save_responses_to_file(query, response_data)
        
        return formatted_responses
    
    def _save_responses_to_file(self, query: str, response_data: Dict):
        """Save responses to a file in the responses directory"""
        try:
            # Create a safe filename from the query
            safe_query = re.sub(r'[^\w\s-]', '', query.lower())
            safe_query = re.sub(r'[\s-]+', '_', safe_query)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_query}_{timestamp}.txt"
            filepath = os.path.join(RESPONSES_DIRECTORY, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for service, response in response_data.items():
                    f.write(f"=== {service.upper()} RESPONSE ===\n")
                    if isinstance(response, ServiceError):
                        f.write(f"Error: {response.message}\n")
                    elif isinstance(response, Exception):
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

class MultiAgentLLM:
    """Main application class"""
    def __init__(self, model_name: str, use_history: bool):
        logger.info(f"Initializing MultiAgentLLM with model={model_name}, use_history={use_history}")
        
        # Check CUDA availability
        device_type = self._check_cuda_availability()
        
        self.api_clients = APIClientManager(device_type=device_type)
        self.llm_manager = LLMManager(model_name, use_history)
        self.response_manager = ResponseManager(self.api_clients, self.llm_manager)
        self.formatter = OutputFormatter()
        
    def _check_cuda_availability(self):
        """Check if CUDA is available and return device information"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Using CPU instead.")
                return "cpu"
            
            # CUDA is available
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"CUDA is available! Found {device_count} CUDA device(s)")
            logger.info(f"Using CUDA device {current_device}: {device_name}")
            
            return "cuda"
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            logger.error("Falling back to CPU mode")
            return "cpu"
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return formatted results"""
        try:
            responses = await self.response_manager.get_responses(query)
            return self.formatter.format_response(responses, query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": {"content": f"Error processing query: {str(e)}", "error": True}}

def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file"""
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    pdf_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return pdf_path

def ingest_pdfs(files):
    """Ingest PDF files into the vector database"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    
    # Use the vector store directory from session state
    persist_dir = st.session_state.vector_store_dir
    os.makedirs(persist_dir, exist_ok=True)
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
        # Use a fixed target dimension of 768 to ensure consistency
        embedding_function = OllamaEmbeddings(target_dimension=768)
        
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_function,
            persist_directory=persist_dir
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
        
    # Initialize vector store directory
    if "vector_store_dir" not in st.session_state:
        st.session_state.vector_store_dir = "./DB/SLAVE"

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
                    # Reset app to use new model
                    st.session_state.app = None
                    st.rerun()
                else:
                    st.error(f"Failed to pull model: {model_name}")

def chat_interface():
    """Chat interface tab"""
    st.title("MultiAgent LLM Chat")
    
    # Initialize app if not already done
    if st.session_state.app is None:
        with st.spinner("Initializing MultiAgent LLM..."):
            try:
                st.session_state.app = MultiAgentLLM(st.session_state.model_name, st.session_state.use_history)
                st.success("MultiAgent LLM initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing MultiAgent LLM: {e}")
                st.warning("Please check your configuration and try again.")
                return
    
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
                    # Run the async function to get responses
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    responses = loop.run_until_complete(st.session_state.app.process_query(prompt))
                    loop.close()
                    
                    # Store responses in session state
                    st.session_state.responses[message_id] = responses
                    
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
                    
                    # Add assistant message to chat history (use local_llm response as default)
                    if "local_llm" in responses and not responses["local_llm"].get("error", False):
                        assistant_response = responses["local_llm"]["content"]
                    else:
                        # Find the first non-error response
                        assistant_response = "I couldn't generate a response at the moment."
                        for service, response in responses.items():
                            if not response.get("error", False):
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
                st.info(f"Database location: {st.session_state.vector_store_dir}")
                
                # Reinitialize app to use the new database
                st.session_state.app = None
                st.rerun()
    
    if st.button("View Database Info"):
        try:
            # Use a fixed target dimension of 768 to ensure consistency
            embedding_function = OllamaEmbeddings(target_dimension=768)
            db = Chroma(
                persist_directory=st.session_state.vector_store_dir,
                embedding_function=embedding_function
            )
            st.write(f"Total documents in database: {db._collection.count()}")
        except Exception as e:
            st.error(f"Error accessing database: {e}")

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

def database_management_interface():
    """Database management tab"""
    st.title("Vector Database Management")
    
    # Vector store folder selector
    st.subheader("Vector Store Folder Selection")
    
    # Get list of existing DB folders
    db_base_dir = "./DB"
    if not os.path.exists(db_base_dir):
        os.makedirs(db_base_dir)
        
    db_folders = ["./DB/SLAVE"]  # Default folder
    
    # List all subdirectories in the DB directory
    try:
        for item in os.listdir(db_base_dir):
            item_path = os.path.join(db_base_dir, item)
            if os.path.isdir(item_path):
                folder_path = f"./DB/{item}"
                if folder_path not in db_folders:
                    db_folders.append(folder_path)
    except Exception as e:
        st.error(f"Error listing DB folders: {e}")
    
    # Custom folder input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Allow user to select from existing folders or enter a new one
        selected_folder = st.selectbox(
            "Select Vector Store Folder",
            db_folders,
            index=db_folders.index(st.session_state.vector_store_dir) if st.session_state.vector_store_dir in db_folders else 0
        )
        
        # Custom folder input
        custom_folder = st.text_input(
            "Or enter a custom folder path",
            placeholder="e.g., ./DB/CUSTOM"
        )
    
    with col2:
        # Apply button for existing folder
        if st.button("Apply Selected", key="apply_selected"):
            if selected_folder != st.session_state.vector_store_dir:
                # Update the session state
                st.session_state.vector_store_dir = selected_folder
                # Reset app to use the new database
                st.session_state.app = None
                st.success(f"Switched to vector store folder: {selected_folder}")
                st.rerun()
        
        # Apply button for custom folder
        if st.button("Apply Custom", key="apply_custom") and custom_folder:
            if not custom_folder.startswith("./DB/"):
                st.error("Custom folder must be within the ./DB/ directory")
            else:
                # Create the directory if it doesn't exist
                if not os.path.exists(custom_folder):
                    os.makedirs(custom_folder)
                # Update the session state
                st.session_state.vector_store_dir = custom_folder
                # Reset app to use the new database
                st.session_state.app = None
                st.success(f"Created and switched to vector store folder: {custom_folder}")
                st.rerun()
    
    # Display current vector store folder
    st.info(f"Current vector store folder: {st.session_state.vector_store_dir}")
    
    with st.expander("Database Information"):
        try:
            # Use default dimension first, then try to detect actual dimension
            embedding_function = OllamaEmbeddings(target_dimension=768)
            db = Chroma(
                persist_directory=st.session_state.vector_store_dir,
                embedding_function=embedding_function
            )
            
            count = db._collection.count()
            st.write(f"Total documents: {count}")
            
            collection_info = db._collection.get()
            if collection_info and 'embeddings' in collection_info and len(collection_info['embeddings']) > 0:
                dimension = len(collection_info['embeddings'][0])
                st.info(f"Embedding dimension: {dimension}")
                
                # Update embedding function to match
                embedding_function = OllamaEmbeddings(target_dimension=dimension)
                
                # Sample a document to show content
                if count > 0 and 'documents' in collection_info:
                    with st.expander("Sample document"):
                        st.write(collection_info['documents'][0][:500] + "...")
            else:
                st.warning("No embeddings found in the database.")
        except Exception as e:
            st.error(f"Error accessing database: {e}")
    
    st.subheader("Reset or Recreate Database")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reset Database", type="primary", help="Deletes the current vector database"):
            try:
                import shutil
                persist_dir = st.session_state.vector_store_dir
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
                    st.success(f"Database at {persist_dir} deleted successfully. Upload new documents to create a fresh database.")
                    
                    # Reinitialize app to use the new database
                    st.session_state.app = None
                    st.rerun()
                else:
                    st.info(f"No database found at {persist_dir}.")
            except Exception as e:
                st.error(f"Error deleting database: {e}")
    
    with col2:
        target_dim = st.number_input("New DB Dimension", 
                                  min_value=64, 
                                  max_value=8192, 
                                  value=768,
                                  help="Set the dimension for a new database")
    
    st.subheader("Database Compatibility Check")
    if st.button("Check Model Compatibility"):
        with st.spinner("Testing embedding dimensions..."):
            try:
                # Test the current model's raw dimension
                embedding_function = OllamaEmbeddings()  # No dimension adjustment
                test_embedding = embedding_function.embed_query("test dimension check")
                raw_dimension = len(test_embedding)
                
                st.info(f"Your current model ({st.session_state.model_name}) produces embeddings with dimension: {raw_dimension}")
                
                # Check if DB exists and its dimension
                persist_dir = st.session_state.vector_store_dir
                if os.path.exists(persist_dir):
                    try:
                        # Use adjusted embeddings to read the DB
                        adjusted_embeddings = OllamaEmbeddings(target_dimension=768)
                        test_db = Chroma(persist_directory=persist_dir, embedding_function=adjusted_embeddings)
                        collection_info = test_db._collection.get()
                        
                        if collection_info and 'embeddings' in collection_info and len(collection_info['embeddings']) > 0:
                            db_dimension = len(collection_info['embeddings'][0])
                            st.info(f"Your database has embeddings with dimension: {db_dimension}")
                            
                            if raw_dimension != db_dimension:
                                st.warning(f"⚠️ Dimension mismatch! The model produces {raw_dimension}-dimensional embeddings " 
                                          f"but your database contains {db_dimension}-dimensional embeddings.")
                                st.info("The automatic dimension adjustment is handling this difference.")
                            else:
                                st.success("✅ Dimensions match! Your model and database are compatible.")
                    except Exception as e:
                        st.error(f"Error checking database dimension: {e}")
                else:
                    st.info("No existing database found. When you create one, it will use adjusted dimensions.")
            except Exception as e:
                st.error(f"Error testing embeddings: {e}")

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="MultiAgent LLM System", layout="wide")
    initialize_session_state()
    
    # Add sidebar with connection information and model selector
    with st.sidebar:
        st.title("MultiAgent LLM System")
        st.caption("Version 7.0")
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

# Import our minimal Anthropic interface
from anthropic_minimal import MinimalAnthropic
import openai
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search
from testcommandprompt2 import OpenSourceApp
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from prompt_template_utils import get_prompt_template
from constants import (
    MODELS_PATH,
    MAX_NEW_TOKENS,
    CHROMA_SETTINGS
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Shared Constants
OLLAMA_HOST = "http://localhost:11434"  # Ollama server base URL
INFERENCE_SERVER_URL = f"{OLLAMA_HOST}/api/chat"  # Ollama chat API endpoint
EMBEDDING_SERVER_URL = f"{OLLAMA_HOST}/api/embeddings"  # Ollama embeddings endpoint
MODEL_NAME = "llama3:8b"  # Default model
EMBEDDING_MODEL_NAME = None  # Will use the same as MODEL_NAME by default
PDF_DIRECTORY = "./pdfs"
# Default vector store directory - will be updated from session state if available
PERSIST_DIRECTORY = "./DB/SLAVE"  # Use local DB directory
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

def handle_service_error(func):
    """Decorator for handling service errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise ServiceError(
                service=func.__name__,
                message="Service temporarily unavailable",
                original_error=e
            )
    return wrapper

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

class OllamaEmbeddings(Embeddings):
    """Embeddings implementation using Ollama's embedding API with dimension handling."""
    
    def __init__(self, model_name=None, target_dimension=None):
        self.model_name = model_name or MODEL_NAME
        self.embedding_url = EMBEDDING_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.target_dimension = target_dimension  # Target dimension for compatibility
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to the embedding server."""
        try:
            # Simple test with a short text
            self.embed_query("test")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama embedding server: {e}")
            return False
    
    def embed_documents(self, texts):
        """Embed multiple documents."""
        return [self.embed_query(text) for text in texts]

    def _adjust_dimension(self, embedding, target_dim):
        """Adjust embedding dimension to match target dimension."""
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate to target dimension
            return embedding[:target_dim]
        else:
            # Pad with zeros to reach target dimension
            return embedding + [0.0] * (target_dim - current_dim)

    def embed_query(self, text):
        """Embed a single text query."""
        # Ollama API format for embeddings
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            # Send request to Ollama API
            response = requests.post(self.embedding_url, json=payload, headers=self.headers)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract embedding from response
            if "embedding" in response_json:
                embedding = response_json["embedding"]
                if isinstance(embedding, list) and len(embedding) > 0:
                    # Adjust dimension if target dimension is specified
                    if self.target_dimension is not None:
                        return self._adjust_dimension(embedding, self.target_dimension)
                    return embedding
                else:
                    raise ValueError(f"Invalid embedding format: {type(embedding)}")
            else:
                raise ValueError(f"Unexpected response format. Keys: {list(response_json.keys())}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with the embedding server: {e}")
            # Return a zero vector as fallback (better than None for some operations)
            dim = self.target_dimension if self.target_dimension is not None else 4096
            return [0.0] * dim
        except ValueError as e:
            logger.error(str(e))
            dim = self.target_dimension if self.target_dimension is not None else 4096
            return [0.0] * dim

class OllamaLLM(LLM):
    """LLM implementation for Ollama API."""
    
    model_name: str = MODEL_NAME
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Ollama API and return the response."""
        headers = {"Content-Type": "application/json"}
        
        # Format messages for Ollama API
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        
        try:
            response = requests.post(INFERENCE_SERVER_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            if "message" in response_json and "content" in response_json["message"]:
                return response_json["message"]["content"]
            else:
                logger.warning("Unexpected response format from LLM server.")
                return "I'm sorry, I couldn't generate a response at the moment."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with the inference server: {e}")
            return "An error occurred while communicating with the LLM server."
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}

class APIClientManager:
    """Manages API clients for different services"""
    def __init__(self, device_type: str = "cuda"):
        Config.validate()
        self.openai_client = self._init_openai()
        self.anthropic_client = self._init_anthropic()
        self.perplexity_client = self._init_perplexity()
        self.gemini_client = self._init_gemini()
        self.wiki = WikipediaAPIWrapper()
        
        # Try to initialize with CUDA, fall back to CPU if it fails
        try:
            logger.info(f"Initializing OpenSourceApp with device: {device_type}")
            self.zuck_bot = OpenSourceApp(device=device_type)
        except RuntimeError as e:
            if "Could not find any implementations for backend: cuda" in str(e) or "libcudart.so" in str(e):
                logger.warning(f"Failed to initialize with CUDA: {e}")
                logger.warning("Falling back to CPU for OpenSourceApp")
                self.zuck_bot = OpenSourceApp(device="cpu")
            else:
                raise
        
    def _init_openai(self):
        try:
            if Config.OPENAI_API_KEY:
                return openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            return None
    
    def _init_anthropic(self):
        try:
            if Config.ANTHROPIC_API_KEY:
                return MinimalAnthropic(api_key=Config.ANTHROPIC_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing Anthropic client: {e}")
            return None
    
    def _init_perplexity(self):
        try:
            if Config.PERPLEXITY_API_KEY:
                return ChatPerplexity(api_key=Config.PERPLEXITY_API_KEY)
            return None
        except Exception as e:
            logger.warning(f"Error initializing Perplexity client: {e}")
            return None
    
    def _init_gemini(self):
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                return genai.GenerativeModel('gemini-pro')
            return None
        except Exception as e:
            logger.warning(f"Error initializing Gemini client: {e}")
            return None

class LLMManager:
    """Manages local LLM operations"""
    def __init__(self, model_name: str, use_history: bool):
        self.model_name = model_name
        self.use_history = use_history
        self.embedding_model_name = EMBEDDING_MODEL_NAME or model_name
        
        logger.info(f"Initializing LLM Manager with model: {model_name}, history: {use_history}")
        
        # Ensure the model is available in Ollama
        ensure_ollama_model(model_name)
        
        # Initialize QA pipeline
        self.qa = self._init_qa_pipeline()
        
    def _init_qa_pipeline(self):
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        
        # Use OllamaEmbeddings
        embeddings = OllamaEmbeddings(model_name=self.embedding_model_name, target_dimension=768)
        
        # Use the vector store directory from session state
        persist_dir = st.session_state.vector_store_dir
        logger.info(f"Connecting to vector database at: {persist_dir}")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            logger.warning(f"Created empty vector database directory at {persist_dir}")
            logger.warning("You may need to ingest documents before querying")
        
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        retriever = db.as_retriever()
        
        logger.info("Setting up prompt template")
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=self.use_history)
        
        logger.info(f"Loading LLM model: {self.model_name}")
        
        # Use OllamaLLM
        llm = OllamaLLM(model_name=self.model_name, temperature=0.7)
        
        qa_kwargs = {
            "llm": llm,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "chain_type_kwargs": {"prompt": prompt}
        }
        
        if self.use_history:
            qa_kwargs["chain_type_kwargs"]["memory"] = memory
        
        logger.info("Initializing QA pipeline")
        return RetrievalQA.from_chain_type(**qa_kwargs)

class ResponseManager:
    """Manages responses from different services"""
    def __init__(self, api_clients: APIClientManager, llm_manager: LLMManager):
        self.clients = api_clients
        self.llm = llm_manager
        
        # Determine which services are available
        available_services = ["local_llm", "wikipedia"]
        if api_clients.openai_client:
            available_services.append("openai")
        if api_clients.anthropic_client:
            available_services.append("anthropic")
        if api_clients.perplexity_client:
            available_services.append("perplexity")
        if api_clients.gemini_client:
            available_services.append("gemini")
            
        logger.info("Response Manager initialized with services: " + ", ".join(available_services))
        
    async def get_responses(self, query: str) -> Dict[str, Any]:
        """Get responses from all services concurrently"""
        logger.info(f"Processing query: {query}")
        
        # Create tasks for all services
        tasks = [
            self._get_local_llm(query),
            self._get_wiki_research(query),
            self._get_chat_output(query)
        ]
        
        # Add optional API services if available
        if self.clients.openai_client:
            tasks.append(self._get_openai_response(query))
        if self.clients.anthropic_client:
            tasks.append(self._get_anthropic_response(query))
        if self.clients.perplexity_client:
            tasks.append(self._get_perplexity_response(query))
        if self.clients.gemini_client:
            tasks.append(self._get_gemini_response(query))
            
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build response dictionary
        response_dict = {
            "local_llm": results[0],
            "wikipedia": results[1],
            "chat": results[2]
        }
        
        # Add optional API responses
        result_index = 3
        if self.clients.openai_client:
            response_dict["openai"] = results[result_index]
            result_index += 1
        if self.clients.anthropic_client:
            response_dict["anthropic"] = results[result_index]
            result_index += 1
        if self.clients.perplexity_client:
            response_dict["perplexity"] = results[result_index]
            result_index += 1
        if self.clients.gemini_client:
            response_dict["gemini"] = results[result_index]
        
        return response_dict
    
    @handle_service_error
    async def _get_local_llm(self, query: str) -> Dict:
        logger.info("Querying local LLM")
        res = self.llm.qa(query)
        return {
            "content": res["result"],
            "sources": res.get("source_documents", [])
        }
    
    @handle_service_error
    async def _get_wiki_research(self, query: str) -> Dict:
        logger.info("Querying Wikipedia")
        return {"content": self.clients.wiki.run(query)}
    
    @handle_service_error
    async def _get_chat_output(self, query: str) -> Dict:
        logger.info("Querying web search")
        try:
            # Use only the parameters that are supported by the search function
            urls = []
            try:
                # Try with no parameters first
                urls = [i for i in search(query)][:5]  # Get first 5 results
            except Exception as e:
                logger.warning(f"Basic search failed: {e}")
                # If that fails, try with just the query
                try:
                    urls = [i for i in search(query)][:5]
                except Exception as e:
                    logger.error(f"All search attempts failed: {e}")
            
            for url in urls:
                try:
                    self.clients.zuck_bot.add("web_page", url)
                except Exception as e:
                    logger.warning(f"Failed to add URL {url}: {e}")
                    continue
            return {"content": self.clients.zuck_bot.query(query)}
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {"content": "Web search unavailable at the moment."}
    
    @handle_service_error
    async def _get_openai_response(self, query: str) -> Dict:
        if self.clients.openai_client is None:
            raise ServiceError(
                service="_get_openai_response",
                message="OpenAI client not available",
                original_error=None
            )
        
        logger.info("Querying OpenAI")
        response = self.clients.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}],
            max_tokens=1000
        )
        return {"content": response.choices[0].message.content}
    
    @handle_service_error
    async def _get_anthropic_response(self, query: str) -> Dict:
        if self.clients.anthropic_client is None:
            raise ServiceError(
                service="_get_anthropic_response",
                message="Anthropic client not available",
                original_error=None
            )
        
        logger.info("Querying Anthropic")
        response = await self.clients.anthropic_client.generate_response(
            prompt=query,
            max_tokens=1000,
            model="claude-3-5-sonnet-latest"
        )
        return {"content": response}
    
    @handle_service_error
    async def _get_perplexity_response(self, query: str) -> Dict:
        if self.clients.perplexity_client is None:
            raise ServiceError(
                service="_get_perplexity_response",
                message="Perplexity client not available",
                original_error=None
            )
        
        logger.info("Querying Perplexity")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{query}")
        ])
        chain = prompt_template | self.clients.perplexity_client
        response = chain.invoke({"query": query})
        return {"content": response.content}
    
    @handle_service_error
    async def _get_gemini_response(self, query: str) -> Dict:
        if self.clients.gemini_client is None:
            raise ServiceError(
                service="_get_gemini_response",
                message="Gemini client not available",
                original_error=None
            )
        
        logger.info("Querying Gemini")
        response = self.clients.gemini_client.generate_content(query)
        return {"content": response.text}

class OutputFormatter:
    """Formats the output for display and saves responses to files"""
    def __init__(self):
        # Create responses directory if it doesn't exist
        if not os.path.exists(RESPONSES_DIRECTORY):
            os.makedirs(RESPONSES_DIRECTORY)
            logger.info(f"Created responses directory at {RESPONSES_DIRECTORY}")
    
    def format_response(self, response_data: Dict, query: str) -> Dict[str, Any]:
        """Format the response for display and save to file"""
        formatted_responses = {}
        
        for service, response in response_data.items():
            if isinstance(response, ServiceError):
                formatted_responses[service] = {
                    "content": f"Error: {response.message}",
                    "error": True
                }
            elif isinstance(response, Exception):
                formatted_responses[service] = {
                    "content": f"Error: {str(response)}",
                    "error": True
                }
            else:
                formatted_responses[service] = {
                    "content": response["content"],
                    "error": False
                }
                if service == "local_llm" and response.get("sources"):
                    formatted_responses[service]["sources"] = response["sources"]
        
        # Save responses to file
        self._save_responses_to_file(query, response_data)
        
        return formatted_responses
    
    def _save_responses_to_file(self, query: str, response_data: Dict):
        """Save responses to a file in the responses directory"""
        try:
            # Create a safe filename from the query
            safe_query = re.sub(r'[^\w\s-]', '', query.lower())
            safe_query = re.sub(r'[\s-]+', '_', safe_query)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_query}_{timestamp}.txt"
            filepath = os.path.join(RESPONSES_DIRECTORY, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for service, response in response_data.items():
                    f.write(f"=== {service.upper()} RESPONSE ===\n")
                    if isinstance(response, ServiceError):
                        f.write(f"Error: {response.message}\n")
                    elif isinstance(response, Exception):
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

class MultiAgentLLM:
    """Main application class"""
    def __init__(self, model_name: str, use_history: bool):
        logger.info(f"Initializing MultiAgentLLM with model={model_name}, use_history={use_history}")
        
        # Check CUDA availability
        device_type = self._check_cuda_availability()
        
        self.api_clients = APIClientManager(device_type=device_type)
        self.llm_manager = LLMManager(model_name, use_history)
        self.response_manager = ResponseManager(self.api_clients, self.llm_manager)
        self.formatter = OutputFormatter()
        
    def _check_cuda_availability(self):
        """Check if CUDA is available and return device information"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Using CPU instead.")
                return "cpu"
            
            # CUDA is available
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"CUDA is available! Found {device_count} CUDA device(s)")
            logger.info(f"Using CUDA device {current_device}: {device_name}")
            
            return "cuda"
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            logger.error("Falling back to CPU mode")
            return "cpu"
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return formatted results"""
        try:
            responses = await self.response_manager.get_responses(query)
            return self.formatter.format_response(responses, query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": {"content": f"Error processing query: {str(e)}", "error": True}}

def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file"""
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    pdf_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return pdf_path

def ingest_pdfs(files):
    """Ingest PDF files into the vector database"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    
    # Use the vector store directory from session state
    persist_dir = st.session_state.vector_store_dir
    os.makedirs(persist_dir, exist_ok=True)
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
        # Use a fixed target dimension of 768 to ensure consistency
        embedding_function = OllamaEmbeddings(target_dimension=768)
        
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_function,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        
        status_text.text(f"Successfully processed {len(all_documents)} document chunks")
        return len(all_documents)
    return 0
