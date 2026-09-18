# AI Student Support Assistant

**Capabilities:** RAG + Tools + Memory

An intelligent College Support Assistant powered by LangChain, Ollama (`llama3.2`), and ChromaDB to answer student queries regarding college regulations, exam patterns, attendance policies, and hostel rules with conversational context awareness.

## Setup & Running

1. Ensure Ollama is running and model `llama3.2` is pulled:
   ```bash
   ollama pull llama3.2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```
