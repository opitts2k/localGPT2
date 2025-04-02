# Modified version that doesn't require gpt4all
import requests
from bs4 import BeautifulSoup
from typing import List
import textwrap
import time

try:
    from googlesearch import search
except ImportError:
    # Mock implementation if googlesearch is not available
    def search(query, **kwargs):
        print(f"Mock search for: {query}")
        return ["https://example.com"]

# Mock GPT4All class
class MockGPT4All:
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device
        print(f"Initialized MockGPT4All with device: {device}")
    
    def generate(self, prompt, max_tokens=512, temp=0.7, top_k=40, top_p=0.9):
        return f"This is a mock response. The GPT4All module is not available. Your query was: {prompt[:100]}..."

# Try to import the real GPT4All, fall back to mock if not available
try:
    from gpt4all import GPT4All
    print("Using real GPT4All")
except ImportError:
    print("GPT4All not available, using mock implementation")
    GPT4All = MockGPT4All

class ChatBot:
    def __init__(self, model_path: str = "Meta-Llama-3-8B-Instruct.Q4_0.gguf", device: str = "cuda"):
        try:
            self.model = GPT4All(model_path, device=device)
        except Exception as e:
            print(f"Error initializing GPT4All: {e}")
            self.model = MockGPT4All(model_path, device=device)
        
        self.history = []
        self.generated = ["Hello! Ask me anything and I'll search and analyze relevant online resources for you."]
        self.past = ["Hey!"]
        self.context_data = []
        
    def search_web(self, query: str, num_results: int = 5) -> List[str]:
        """Perform Google search and return URLs"""
        try:
            # Add search modifiers to focus on reliable sources
            search_query = f"{query} site:scholar.google.com OR site:.edu OR site:.gov"
            urls = [url for url in search(
                search_query,
                tld="co.in",
                num=num_results,
                stop=num_results,
                pause=2  # Be nice to Google's servers
            )]
            return urls
        except Exception as e:
            print(f"Search error: {e}")
            return ["https://example.com/mock-result"]

    def process_webpage(self, url: str, max_length: int = 32000) -> str:
        """Fetch and process webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            return text[:max_length]
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return f"Mock content for {url}"
    
    def chunk_text(self, text: str, chunk_size: int = 8000) -> List[str]:
        """Split text into manageable chunks"""
        return textwrap.wrap(text, chunk_size, break_long_words=False, break_on_hyphens=False)
    
    def create_prompt(self, context: str, query: str) -> str:
        """Create a prompt with context"""
        return f"""Context: {context}

Question: {query}

Based on the above context, please provide a comprehensive answer to the question. If the context doesn't contain relevant information, please indicate that."""
    
    def query(self, user_input: str) -> str:
        """Process query by searching, analyzing content, and generating response"""
        try:
            # Clear previous context for new query
            self.context_data = []
            
            # First, search for relevant URLs
            print("Searching for relevant sources...")
            urls = self.search_web(user_input)
            if not urls:
                return "I couldn't find any relevant sources. Please try reformulating your query."
            
            # Process each URL
            print(f"Analyzing {len(urls)} sources...")
            for url in urls:
                content = self.process_webpage(url)
                if content:
                    self.context_data.append(content)
                time.sleep(0.5)  # Polite delay between requests
            
            if not self.context_data:
                return "I found some sources but couldn't extract useful content. Please try a different query."
            
            # Process query against all context chunks
            print("Generating response...")
            combined_text = " ".join(self.context_data)
            chunks = self.chunk_text(combined_text)
            
            responses = []
            for chunk in chunks:
                try:
                    prompt = self.create_prompt(chunk, user_input)
                    response = self.model.generate(prompt, max_tokens=512)
                    responses.append(response)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    responses.append(f"Error processing chunk: {str(e)}")
            
            final_response = "\n\n".join(responses)
            return final_response or "I couldn't generate a meaningful response from the sources."
            
        except Exception as e:
            return f"An error occurred while processing your query: {str(e)}"
    
    def add(self, source_type, content):
        """Add content to the context data"""
        if source_type == "web_page":
            try:
                webpage_content = self.process_webpage(content)
                if webpage_content:
                    self.context_data.append(webpage_content)
                    return True
            except Exception as e:
                print(f"Error adding web page: {e}")
        return False
    
    def process_input(self, user_input: str) -> str:
        try:
            # Add input to history
            self.past.append(user_input)
            
            # Generate response using the model
            try:
                response = self.model.generate(
                    user_input,
                    max_tokens=512,
                    temp=0.7,
                    top_k=40,
                    top_p=0.9
                )
            except Exception as e:
                response = f"Error generating response: {str(e)}"
            
            # Add response to history
            self.generated.append(response)
            return response
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.generated.append(error_msg)
            return error_msg
    
    def get_history(self):
        """Return chat history"""
        return list(zip(self.past, self.generated))

def OpenSourceApp(device: str = "cuda"):
    """Function to create and return a ChatBot instance"""
    return ChatBot(device=device)

if __name__ == "__main__":
    main = lambda: None  # Placeholder to avoid running the main function
