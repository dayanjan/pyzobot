# PyZoBot

![PyZoBot Logo](https://i.postimg.cc/4xPdhkB2/PYZo-Bot-new-logo-small.png)

PyZoBot is a powerful platform that enables conversational information extraction and synthesis from curated Zotero reference libraries through advanced retrieval-augmented generation. It combines the comprehensive reference management capabilities of Zotero with state-of-the-art language models to provide an intuitive interface for querying and analyzing academic literature.

## Features

- **Zotero Integration**: Direct connection to Zotero libraries (both personal and group libraries)
- **Advanced Document Processing**: Automatic PDF extraction and processing from Zotero attachments
- **Intelligent Chunking**: Smart text splitting with customizable chunk sizes and overlap
- **Vector Search**: Utilizes OpenAI embeddings for semantic search capabilities
- **Contextual Compression**: Employs retrieval-augmented generation for more accurate responses
- **Citation Support**: Automatically includes citations and references in responses
- **Chat Interface**: User-friendly chat interface with downloadable chat history
- **Customizable Parameters**: Adjustable settings for model selection, token limits, and document retrieval

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Zotero API key
- Zotero library ID (user or group)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pyzobot.git
cd pyzobot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Required Dependencies

The following key packages are required (see requirements.txt for complete list):

- streamlit==1.38.0
- pyzotero==1.5.18
- pandas==2.0.0
- llama_index==0.11.7
- langchain==0.2.16
- openai==1.44.0
- chromadb==0.4.15
- PyMuPDF==1.24.10
- PyPDF2==3.0.1

## Usage

1. Run the Streamlit app:
```bash
streamlit run PyZoBot_v18.py
```

2. Input your configuration in the sidebar:
   - OpenAI API Key
   - Zotero API Key
   - Library Type (group/user)
   - Library ID
   - Chunk Size and Overlap settings
   - Model parameters (GPT-4/GPT-3.5-turbo)

3. Click "Fetch PDFs from Zotero" to download and process your library

4. Start asking questions in the chat interface

## Features in Detail

### Document Processing
- Automatically fetches PDFs from your Zotero library
- Processes documents using PyMuPDF for text extraction
- Implements recursive character text splitting for optimal chunking

### Vector Store
- Creates embeddings using OpenAI's text-embedding-ada-002 model
- Stores vectors using Chroma for efficient retrieval
- Supports MMR (Maximum Marginal Relevance) search for diverse results

### Question Answering
- Utilizes GPT-4 or GPT-3.5-turbo for response generation
- Implements contextual compression for more relevant results
- Provides source citations and references for transparency

### User Interface
- Clean, intuitive Streamlit interface
- Real-time response generation
- Downloadable chat history
- Adjustable parameters for customization

## Configuration Options

- **Chunk Size**: Controls the length of text segments (100-5000 tokens)
- **Chunk Overlap**: Determines overlap between adjacent chunks (0-5000 tokens)
- **Model Selection**: Choice between GPT-4 and GPT-3.5-turbo
- **Max Tokens**: Limit for response length (100-4000 tokens)
- **Number of Documents**: Control how many documents to retrieve (1-30)

## Best Practices

1. Start with default chunking parameters and adjust based on your needs
2. Use GPT-4 for more complex queries requiring detailed analysis
3. Adjust the number of retrieved documents based on the complexity of your questions
4. Download chat history regularly to save important conversations

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**:
   - Ensure both OpenAI and Zotero API keys are correct
   - Check if you have sufficient API credits

2. **PDF Processing Issues**:
   - Verify PDF permissions in Zotero
   - Ensure PDFs are properly attached in your Zotero library

3. **Memory Issues**:
   - Reduce chunk size for large libraries
   - Process fewer documents at once

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use PyZoBot in your research, please cite:

```
Alshammari, S., Abu Rukbah, W., Basalelah, L., Alsuhibani, A., Alghubayshi, A., & Wijesinghe, D. S. (2024). 
PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Zotero team for their excellent reference management system
- OpenAI for their powerful language models
- The LangChain and LlamaIndex communities

## Contact

For questions and support, please [open an issue](https://github.com/yourusername/pyzobot/issues) on our GitHub repository.
