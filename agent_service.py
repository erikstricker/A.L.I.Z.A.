"""Agent service module for MCP-powered AI agents.

This module provides an MCPAgent class that integrates with Model Context Protocol
(MCP) servers to create AI agents with enhanced tool capabilities.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

class MCPAgent:
    """AI agent that integrates with Model Context Protocol servers.
    
    This agent connects to MCP servers to access specialized tools and capabilities,
    then uses LangGraph's ReAct pattern to process user queries intelligently.
    
    Attributes:
        openai_api_key: OpenAI API key for LLM access
        server_url: URL of the MCP server to connect to
        agent: The initialized LangGraph agent instance
    """
    
    def __init__(self, openai_api_key: str, server_url: str) -> None:
        """Initialize the MCPAgent with API key and server URL.
        
        Args:
            openai_api_key: Valid OpenAI API key
            server_url: URL endpoint of the MCP server
        """
        self.openai_api_key = openai_api_key
        self.server_url = server_url
        self.agent: Optional[Any] = None
        
    async def initialize(self) -> None:
        """Initialize the agent with MCP client and tools.
        
        This method:
        1. Sets up the OpenAI API key in environment
        2. Creates MCP client connection to the server
        3. Retrieves available tools from MCP servers
        4. Initializes LLM and creates ReAct agent
        
        Raises:
            Exception: If MCP client connection or tool retrieval fails
        """
        # Configure OpenAI API access
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Configure MCP client with server connection
        mcp_config = {
            "theme": {
                "url": self.server_url,
                "transport": "streamable_http",
            }
        }
        client = MultiServerMCPClient(mcp_config)
        
        # Retrieve available tools from MCP servers
        tools = await client.get_tools()
        
        # Initialize language model with optimal settings for agent use
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0  # Use deterministic responses for consistency
        )
        
        # Create ReAct (Reasoning + Acting) agent with tools
        self.agent = create_react_agent(llm, tools)
    
    async def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Process user messages through the agent.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The agent's response content as a string
            
        Note:
            If agent is not initialized, this method will initialize it first.
        """
        # Ensure agent is initialized before processing
        if not self.agent:
            await self.initialize()
            
        try:
            # Process messages through the agent
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract and return the latest response content
            return response["messages"][-1].content
            
        except Exception as e:
            # Return user-friendly error message
            return f"❌ Error with AI agent: {str(e)}"

# Global agent instance for singleton pattern
_global_agent: Optional[MCPAgent] = None


async def get_agent(openai_api_key: str, server_url: str) -> MCPAgent:
    """Get or create a singleton agent instance.
    
    This function implements a singleton pattern to ensure only one agent
    instance exists throughout the application lifecycle.
    
    Args:
        openai_api_key: OpenAI API key for the agent
        server_url: MCP server URL to connect to
        
    Returns:
        Initialized MCPAgent instance
        
    Note:
        The agent will be created and initialized on first call.
        Subsequent calls return the existing instance.
    """
    global _global_agent
    
    if _global_agent is None:
        _global_agent = MCPAgent(openai_api_key, server_url)
        await _global_agent.initialize()
        
    return _global_agent