"""LangChain Helper Functions.

Centralized AI/ML functionality for the LLM Bootcamp Project.
Provides helper classes for different types of chatbot functionality:
- BasicChatbotHelper: Simple conversational AI
- AgentChatbotHelper: AI with web search capabilities  
- RAGHelper: Retrieval-Augmented Generation for documents
- MCPHelper: Model Context Protocol integration
- ValidationHelper: Input validation utilities
"""

import os
from typing import List, Dict, Any, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END


class BasicChatbotHelper:
    """Helper class for basic conversational chatbot functionality.
    
    Provides utilities to create and manage simple AI chatbots with
    customizable response styles and conversation memory.
    """
    
    @staticmethod
    def build_chain(config: Dict[str, Any], api_key: str = None) -> Any:
        """Build a LangChain chain for basic chatbot functionality.
        
        Creates a conversational chain with customizable LLM parameters
        and response styles (Professional, Casual, Creative, Technical, Balanced).
        
        Args:
            config: Configuration dictionary with model settings
            api_key: Optional OpenAI API key override
            
        Returns:
            Configured LangChain chain ready for conversation
        """
        llm_kwargs = {
            "model": config["model"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "top_p": config.get("top_p", 1.0),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0),
            "streaming": False
        }
        
        if api_key:
            llm_kwargs["api_key"] = api_key
            
        llm = ChatOpenAI(**llm_kwargs)
        
        # Configure response style with predefined system prompts
        system_prompts = {
            "Professional": "You are a professional AI assistant. Provide formal, detailed, and well-structured responses suitable for business contexts.",
            "Casual": "You are a friendly and casual AI assistant. Use conversational language and be approachable in your responses.",
            "Creative": "You are a creative AI assistant. Provide imaginative, engaging responses with varied perspectives and creative insights.",
            "Technical": "You are a technical AI assistant. Provide precise, detailed explanations with technical accuracy and clarity.",
            "Balanced": config.get("system_prompt", "You are a helpful AI assistant. Provide clear, concise, and friendly responses.")
        }
        
        system_message = system_prompts.get(config.get("response_style", "Balanced"), system_prompts["Balanced"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        return prompt | llm
    
    @staticmethod
    def invoke_with_memory(chain: Any, user_input: str, chat_history: List[Dict[str, str]]) -> Any:
        """Invoke the chain with conversation memory support.
        
        Processes user input while maintaining context from previous messages
        in the conversation history.
        
        Args:
            chain: The LangChain chain to invoke
            user_input: Current user message
            chat_history: List of previous conversation messages
            
        Returns:
            Chain response with conversation context
        """
        # Convert chat history to LangChain message format
        formatted_history = []
        for msg in chat_history[:-1]:  # Exclude the current user message
            if msg["role"] == "user":
                formatted_history.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(("assistant", msg["content"]))
        
        return chain.invoke({
            "input": user_input,
            "chat_history": formatted_history
        })
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for basic chatbot.
        
        Returns:
            Dictionary with default model settings optimized for conversation
        """
        return {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000
        }


class AgentChatbotHelper:
    """Helper class for agent chatbot with real-time web search functionality.
    
    Integrates with Tavily search API to provide AI agents with the ability
    to search and retrieve current information from the web.
    """
    
    @staticmethod
    def setup_agent(openai_api_key: str, tavily_api_key: str) -> Any:
        """Setup an AI agent with Tavily web search capabilities.
        
        Creates a ReAct agent that can search the web for real-time information
        to supplement its responses.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
            tavily_api_key: Tavily API key for web search functionality
            
        Returns:
            Configured LangGraph ReAct agent with search tools
        """
        from langchain_tavily import TavilySearch
        from langgraph.prebuilt import create_react_agent
        
        # Configure Tavily search tool with optimal settings
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",
            tavily_api_key=tavily_api_key,
        )
        
        tools = [tavily_search]
        
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=openai_api_key)
        agent = create_react_agent(llm, tools)
        return agent
    
    @staticmethod
    async def process_agent_response(agent: Any, user_query: str) -> str:
        """Process agent response with streaming support.
        
        Handles the agent's reasoning and tool usage steps, collecting
        all output into a cohesive response.
        
        Args:
            agent: The configured LangGraph agent
            user_query: User's question or request
            
        Returns:
            Complete agent response as a string
        """
        accumulated_response = ""
        
        # Stream state updates (includes reasoning and tool execution steps)
        for update in agent.stream({"messages": user_query}):
            messages = update.get("messages", [])
            for message in messages:
                content = getattr(message, "content", "")
                
                # Handle structured content (list of content blocks)
                if not content and isinstance(getattr(message, "content", None), list):
                    content = "".join(
                        block.get("text", "")
                        for block in message.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    
                if content:
                    accumulated_response += content

        # Fallback to direct invocation if streaming failed
        if not accumulated_response:
            response = agent.invoke({"messages": user_query})
            accumulated_response = (
                response["messages"][-1].content
                if isinstance(response, dict) and response.get("messages")
                else str(response)
            )
        
        return accumulated_response


class RAGHelper:
    """Helper class for RAG (Retrieval-Augmented Generation) functionality.
    
    Provides document processing, vector storage, and intelligent retrieval
    capabilities for question-answering over user documents.
    """
    
    # Type definition for RAG workflow state management
    class RAGState(TypedDict):
        question: str
        mode: Literal["summary", "fact"]
        documents: List[Document]
        generation: str
    
    @staticmethod
    def save_file(file, folder: str = "tmp") -> str:
        """Save uploaded file to local storage.
        
        Args:
            file: Streamlit uploaded file object
            folder: Directory to save the file (created if doesn't exist)
            
        Returns:
            Full path to the saved file
        """
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path
    
    @staticmethod
    def build_vectorstore(files, api_key: str = None) -> FAISS:
        """Build FAISS vector store from uploaded PDF files.
        
        Processes PDF files, splits them into chunks, creates embeddings,
        and builds a searchable vector database.
        
        Args:
            files: List of uploaded PDF files
            api_key: Optional OpenAI API key for embeddings
            
        Returns:
            Configured FAISS vector store ready for similarity search
        """
        documents: List[Document] = []
        
        # Process each uploaded PDF file
        for file in files:
            file_path = RAGHelper.save_file(file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        
        # Split documents into manageable chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(documents)
        
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["api_key"] = api_key
            
        # Create embeddings and build vector store
        embeddings = OpenAIEmbeddings(**embeddings_kwargs)
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        
        return vector_store
    
    @staticmethod
    def build_simple_agentic_rag(retriever, llm: ChatOpenAI):
        """Build an intelligent agentic RAG workflow.
        
        Creates a graph-based workflow that automatically determines whether
        to provide summaries or specific facts based on the query type.
        
        Args:
            retriever: Vector store retriever for document search
            llm: Language model for generating responses
            
        Returns:
            Compiled LangGraph workflow for intelligent document QA
        """
        
        # Classification node: determine if query needs summary or specific facts
        SUMMARY_HINTS = ("summarize", "summary", "overview", "key points", "bullet", "synthesize")
        FACT_HINTS = ("when", "date", "who", "where", "amount", "total", "price", "figure", "specific", "exact")
        
        def classify_mode(state: RAGHelper.RAGState) -> RAGHelper.RAGState:
            """Classify query type to determine appropriate response mode."""
            query_lower = state["question"].lower()
            
            # Determine response mode based on query keywords
            if any(hint in query_lower for hint in SUMMARY_HINTS) and not any(hint in query_lower for hint in FACT_HINTS):
                mode: Literal["summary", "fact"] = "summary"
            elif any(hint in query_lower for hint in FACT_HINTS):
                mode = "fact"
            else:
                # Default to summary for general questions, fact for specific queries
                mode = "summary" if "summary" in query_lower or "summarize" in query_lower else "fact"
                
            return {**state, "mode": mode}
        
        # Retrieval node: fetch relevant documents based on query type
        def retrieve(state: RAGHelper.RAGState) -> RAGHelper.RAGState:
            """Retrieve relevant documents based on query and mode."""
            question = state["question"]
            
            # Adjust retrieval count based on response mode
            num_docs = 8 if state["mode"] == "summary" else 3
            retrieved_docs = retriever.invoke(question)
            
            return {**state, "documents": retrieved_docs[:num_docs]}
        
        # Generation node: create appropriate response based on mode and context
        gen_prompt_summary = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Create a concise, faithful summary ONLY using the provided context. "
             "Prefer bullet points if helpful. Do not use outside knowledge."),
            ("human",
             "Question:\n{question}\n\n"
             "Context (multiple document chunks):\n{context}\n\n"
             "Write a grounded summary:")
        ])
        
        gen_prompt_fact = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer precisely and ONLY using the provided context. "
             "If the context is insufficient, say so."),
            ("human",
             "Question:\n{question}\n\n"
             "Context:\n{context}\n\n"
             "Answer:")
        ])
        
        def generate(state: RAGHelper.RAGState) -> RAGHelper.RAGState:
            """Generate response based on retrieved documents and mode."""
            # Combine retrieved document content
            document_context = "\n\n---\n\n".join(
                doc.page_content for doc in state.get("documents", [])
            )
            
            # Handle case where no relevant documents found
            if not document_context.strip():
                return {
                    **state, 
                    "generation": "I couldn't find enough information in the documents to answer that."
                }
            
            # Generate response using appropriate prompt based on mode
            if state["mode"] == "summary":
                response = llm.invoke(gen_prompt_summary.format_messages(
                    question=state["question"], 
                    context=document_context
                ))
            else:
                response = llm.invoke(gen_prompt_fact.format_messages(
                    question=state["question"], 
                    context=document_context
                ))
                
            return {**state, "generation": response.content}
        
        # Construct the workflow graph with connected nodes
        graph = StateGraph(RAGHelper.RAGState)
        graph.add_node("classify_mode", classify_mode)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate)
        
        graph.set_entry_point("classify_mode")
        graph.add_edge("classify_mode", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        
        return graph.compile()
    
    @staticmethod
    def setup_rag_system(uploaded_files, api_key: str = None) -> Any:
        """Setup complete RAG system from uploaded files.
        
        Orchestrates the entire RAG pipeline: file processing, vectorization,
        retriever setup, and workflow creation.
        
        Args:
            uploaded_files: List of PDF files to process
            api_key: Optional OpenAI API key
            
        Returns:
            Complete RAG workflow ready for query processing
        """
        # Build vector store and configure retriever
        vector_store = RAGHelper.build_vectorstore(uploaded_files, api_key)
        retriever = vector_store.as_retriever()
        
        # Configure language model for generation
        llm_config = {
            "model": "gpt-4o-mini", 
            "temperature": 0,  # Deterministic responses
            "streaming": False
        }
        if api_key:
            llm_config["api_key"] = api_key
            
        llm = ChatOpenAI(**llm_config)
        return RAGHelper.build_simple_agentic_rag(retriever, llm)


class MCPHelper:
    """Helper class for Model Context Protocol (MCP) functionality.
    
    Provides utilities for working with MCP-powered agents that can access
    specialized tools and resources beyond standard LLM capabilities.
    """
    
    @staticmethod
    async def get_agent(openai_api_key: str, mcp_server_url: str):
        """Get or create an MCP agent instance.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
            mcp_server_url: URL of the MCP server to connect to
            
        Returns:
            Initialized MCP agent ready for use
        """
        from agent_service import get_agent
        return await get_agent(openai_api_key, mcp_server_url)
    
    @staticmethod
    async def process_mcp_query(agent: Any, messages: List[Dict[str, str]]) -> str:
        """Process a query through the MCP agent.
        
        Args:
            agent: The MCP agent instance
            messages: List of conversation messages
            
        Returns:
            Agent response as a string, or error message if processing fails
        """
        try:
            response_text = await agent.invoke(messages)
            return response_text
        except Exception as e:
            return f"❌ MCP Agent Error: {str(e)}"


class ValidationHelper:
    """Helper class for input validation.
    
    Provides validation utilities for API keys and configuration values
    used throughout the application.
    """
    
    @staticmethod
    def validate_openai_key(api_key: str) -> bool:
        """Validate OpenAI API key format.
        
        Args:
            api_key: API key string to validate
            
        Returns:
            True if key format is valid, False otherwise
        """
        return api_key and api_key.startswith("sk-")
    
    @staticmethod
    def validate_tavily_key(api_key: str) -> bool:
        """Validate Tavily API key format.
        
        Args:
            api_key: Tavily API key string to validate
            
        Returns:
            True if key format is valid, False otherwise
        """
        return api_key and api_key.startswith("tvly-")
    
    @staticmethod
    def validate_mcp_url(url: str) -> bool:
        """Validate MCP server URL format.
        
        Args:
            url: MCP server URL to validate
            
        Returns:
            True if URL format is valid, False otherwise
        """
        return url and (url.startswith("http://") or url.startswith("https://"))