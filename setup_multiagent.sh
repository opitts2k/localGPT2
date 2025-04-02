#!/bin/bash
echo "Setting up MultiAgent LLM environment..."

# Create required directories
mkdir -p ./DB/SLAVE
mkdir -p ./pdfs
mkdir -p ./llmresponses

# Install Python dependencies
pip install -r multiagent_requirements.txt

# Download default model
ollama pull llama3:8b

echo "Setup complete!"
echo "To run the application: streamlit run run_localGPTMultiAgent7_streamlined.py"
