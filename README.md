# AI Chatbot Platform

An enterprise-grade conversational AI platform featuring intelligent agents, document processing, web search integration, and adaptive prompt optimization through Model Context Protocol.

## 🚀 Quick Start

```bash
cd project_code/
pip install -r requirements.txt
streamlit run Home.py
```

## 📁 System Architecture

```
├── project_code/          # Core application
│   ├── .streamlit/        # Configuration
│   ├── assets/            # Static resources
│   ├── pages/             # Interface modules
│   ├── tmp/               # Processing workspace
│   ├── Home.py            # Application entry point
│   ├── server.py          # MCP optimization server
│   ├── requirements.txt   # Dependencies
│   ├── ui_components.py   # Interface components
│   ├── langchain_helpers.py # AI integration layer
│   └── agent_service.py   # Agent orchestration
├── Exercises/             # Implementation guides
├── User_Guide/            # Operation manuals
├── docs/                  # Technical specifications
└── README.md              # Documentation
```

## Platform Capabilities

### Built-in Applications
1. **Basic Chatbot** - Dialogue system with context memory and customizable personalities
2. **Agent Chatbot** - Web-enabled research and real-time information retrieval 
3. **Document Chat** - PDF document analysis and question-answering with RAG
4. **MCP Agent** - Model Context Protocol integration with tool access

### Exercise Implementations
1. **Basic Chatbot** - Conversational AI with customizable personalities and response styles
2. **Chatbot Agent** - Web search integration using Tavily API for real-time information
3. **Chat with your Data** - Multi-format document processing (PDF, MD, HTML) with agentic RAG
4. **MCP Prompt Optimizer** - Feedback-driven prompt optimization using LangMem with persistent storage


## Deployment

1. **System Setup**
   ```bash
   git clone <repository>
   cd llmbootcamp-project/project_code/
   pip install -r requirements.txt
   ```

2. **Configuration**
   - OpenAI API key for language models
   - Tavily API key for web search
   - MCP server endpoint configuration

3. **Launch**
   ```bash
   streamlit run Home.py
   ```

4. **Access**
   - Navigate through 4 different AI interfaces
   - Configure agents for specific use cases
   - Upload documents for analysis