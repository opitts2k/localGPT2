import os
import logging
import asyncio
import click
import sys
from functools import wraps
from typing import Dict, List, Any
from dotenv import load_dotenv

# Import torch after other imports to avoid conflicts
import torch

import anthropic
import openai
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search
from testcommandprompt2 import OpenSourceApp
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline, GenerationConfig
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    MODELS_PATH,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MAX_NEW_TOKENS,
    CHROMA_SETTINGS
)

from prompt_template_utils import get_prompt_template
from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check CUDA availability
def check_cuda_availability():
    """Check if CUDA is available and return device information"""
    try:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This script requires CUDA to run.")
            logger.error("Please ensure you have a CUDA-capable GPU and the required drivers installed.")
            
            # Print detailed system information for debugging
            logger.error(f"PyTorch version: {torch.__version__}")
            logger.error(f"System platform: {sys.platform}")
            
            # Ask user if they want to continue with CPU instead
            print("\n⚠️ CUDA is not available on your system.")
            print("This script is designed to run optimally with CUDA-enabled GPUs.")
            choice = input("Do you want to continue using CPU instead? (y/n): ").strip().lower()
            
            if choice == 'y' or choice == 'yes':
                logger.warning("Continuing with CPU. Performance may be significantly slower.")
                return "cpu"
            else:
                logger.info("Exiting as requested.")
                sys.exit(1)
        
        # CUDA is available
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        logger.info(f"CUDA is available! Found {device_count} CUDA device(s)")
        logger.info(f"Using CUDA device {current_device}: {device_name}")
        
        # Additional CUDA information
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        logger.info(f"CUDA Version: {cuda_version}")
        
        return "cuda"
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {e}")
        logger.error("Falling back to CPU mode")
        return "cpu"

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
            return openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            return None
    
    def _init_anthropic(self):
        try:
            return anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        except TypeError as e:
            logger.warning(f"Error initializing Anthropic client: {e}")
            # Return a placeholder that will be handled by the error handler
            return None
    
    def _init_perplexity(self):
        try:
            return ChatPerplexity(api_key=Config.PERPLEXITY_API_KEY)
        except Exception as e:
            logger.warning(f"Error initializing Perplexity client: {e}")
            return None
    
    def _init_gemini(self):
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.warning(f"Error initializing Gemini client: {e}")
            return None

class LLMManager:
    """Manages local LLM operations"""
    def __init__(self, device_type: str, use_history: bool):
        self.device_type = device_type
        self.use_history = use_history
        logger.info(f"Initializing LLM Manager with device: {device_type}, history: {use_history}")
        self.qa = self._init_qa_pipeline()
        
    def _init_qa_pipeline(self):
        logger.info(f"Loading embeddings model: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        logger.info(f"Connecting to vector database at: {PERSIST_DIRECTORY}")
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        retriever = db.as_retriever()
        
        logger.info("Setting up prompt template")
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=self.use_history)
        
        logger.info(f"Loading LLM model: {MODEL_ID}")
        llm = self._load_model()
        
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
    
    def _load_model(self):
        try:
            if MODEL_BASENAME:
                if ".gguf" in MODEL_BASENAME.lower() or ".ggml" in MODEL_BASENAME.lower():
                    logger.info(f"Loading quantized GGUF/GGML model: {MODEL_BASENAME}")
                    try:
                        return load_quantized_model_gguf_ggml(MODEL_ID, MODEL_BASENAME, self.device_type, logging)
                    except RuntimeError as e:
                        if "Could not find any implementations for backend: cuda" in str(e) or "libcudart.so" in str(e):
                            logger.warning(f"Failed to load model with CUDA: {e}")
                            logger.warning(f"Falling back to CPU for model loading")
                            return load_quantized_model_gguf_ggml(MODEL_ID, MODEL_BASENAME, "cpu", logging)
                        else:
                            raise
                else:
                    logger.info(f"Loading quantized GPTQ model: {MODEL_BASENAME}")
                    try:
                        return load_quantized_model_qptq(MODEL_ID, MODEL_BASENAME, self.device_type, logging)
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"Failed to load model with CUDA: {e}")
                            logger.warning(f"Falling back to CPU for model loading")
                            return load_quantized_model_qptq(MODEL_ID, MODEL_BASENAME, "cpu", logging)
                        else:
                            raise
            else:
                logger.info(f"Loading full model: {MODEL_ID}")
                try:
                    model, tokenizer = load_full_model(MODEL_ID, None, self.device_type, logging)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.warning(f"Failed to load model with CUDA: {e}")
                        logger.warning(f"Falling back to CPU for model loading")
                        model, tokenizer = load_full_model(MODEL_ID, None, "cpu", logging)
                    else:
                        raise
                
                logger.info("Creating text generation pipeline")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=MAX_NEW_TOKENS,
                    temperature=0.2,
                    repetition_penalty=1.15,
                    generation_config=GenerationConfig.from_pretrained(MODEL_ID)
                )
                return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("This is a critical error. Please check your model configuration.")
            raise

class ResponseManager:
    """Manages responses from different services"""
    def __init__(self, api_clients: APIClientManager, llm_manager: LLMManager):
        self.clients = api_clients
        self.llm = llm_manager
        logger.info("Response Manager initialized with services: " + 
                   " ".join([s for s in ["anthropic", "gemini", "local_llm", "wikipedia", 
                                         "openai" if api_clients.openai_client else None,
                                         "perplexity" if api_clients.perplexity_client else None]
                             if s is not None]))
        
    async def get_responses(self, query: str) -> Dict[str, Any]:
        """Get responses from all services concurrently"""
        logger.info(f"Processing query: {query}")
        tasks = [
            self._get_local_llm(query),
            self._get_wiki_research(query),
            self._get_chat_output(query),
            self._get_openai_response(query),
            self._get_anthropic_response(query),
            self._get_perplexity_response(query),
            self._get_gemini_response(query)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "local_llm": results[0],
            "wikipedia": results[1],
            "chat": results[2],
            "openai": results[3],
            "anthropic": results[4],
            "perplexity": results[5],
            "gemini": results[6]
        }
    
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
        urls = [i for i in search(query, num=5, stop=5, pause=2)]
        for url in urls:
            try:
                self.clients.zuck_bot.add("web_page", url)
            except Exception:
                continue
        return {"content": self.clients.zuck_bot.query(query)}
    
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
        response = self.clients.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": query}]
        )
        return {"content": response.content[0].text}
    
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
    """Formats the output for display"""
    @staticmethod
    def format_response(response_data: Dict) -> str:
        output = []
        
        for service, response in response_data.items():
            if isinstance(response, ServiceError):
                output.append(f"> {service.title()} Response: {response.message}")
            else:
                output.append(f"> {service.title()} Response: {response['content']}")
                if service == "local_llm" and response.get("sources"):
                    output.append("\n--- Source Documents ---")
                    for source in response["sources"]:
                        output.append(f"\nSource: {source.metadata.get('source', 'Unknown')}")
                        output.append(source.page_content)
                    output.append("--- End of Sources ---")
        
        return "\n".join(output)

class MultiAgentLLM:
    """Main application class"""
    def __init__(self, device_type: str, use_history: bool):
        logger.info(f"Initializing MultiAgentLLM with device_type={device_type}, use_history={use_history}")
        self.api_clients = APIClientManager(device_type=device_type)
        self.llm_manager = LLMManager(device_type, use_history)
        self.response_manager = ResponseManager(self.api_clients, self.llm_manager)
        self.formatter = OutputFormatter()
        
    async def process_query(self, query: str) -> str:
        """Process a query and return formatted results"""
        try:
            responses = await self.response_manager.get_responses(query)
            return self.formatter.format_response(responses)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

@click.command()
@click.option("--use_history", 
              is_flag=True, 
              help="Use conversation history")
@click.option("--save_responses",
              is_flag=True,
              help="Save responses to a file")
def main(use_history: bool, save_responses: bool):
    """
    MultiAgent LLM System that exclusively uses CUDA for processing.
    This script requires a CUDA-capable GPU to run.
    """
    # Check CUDA availability - will exit if not available
    device_type = check_cuda_availability()
    
    logger.info(f"Device type: {device_type}")
    logger.info(f"Using history: {use_history}")
    logger.info(f"Save responses: {save_responses}")
    
    # Ensure models directory exists
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    
    app = MultiAgentLLM(device_type, use_history)
    
    print("\n🤖 MultiAgent LLM System")
    print("Enter 'exit' to quit, 'help' for commands\n")
    
    responses_file = None
    if save_responses:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        responses_file = open(f"responses_{timestamp}.txt", "w")
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            break
        elif query.lower() == "help":
            print("\nCommands:")
            print("  exit - Exit the program")
            print("  help - Show this help message")
            continue
            
        if query:
            result = asyncio.run(app.process_query(query))
            print(f"\n{result}\n")
            
            if save_responses and responses_file:
                responses_file.write(f"Query: {query}\n\n")
                responses_file.write(f"{result}\n\n")
                responses_file.write("-" * 80 + "\n\n")
                responses_file.flush()
        else:
            print("Please enter a valid query")
    
    if save_responses and responses_file:
        responses_file.close()
        print(f"Responses saved to {responses_file.name}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
