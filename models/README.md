# Model Configuration & Specifications

- **LLM Base Model**: `llama3.2` (Meta Llama 3.2 3B Instruct)
  - **Runtime**: Ollama local server (`http://localhost:11434`)
  - **Architecture**: Dense Autoregressive Transformer
  - **Context Window**: 128k tokens
  - **Quantization**: 4-bit Medium (`Q4_K_M`)
- **Embedding Model**: `nomic-embed-text`
  - **Architecture**: 768-dimensional text embedding model optimized for local RAG retrieval.
