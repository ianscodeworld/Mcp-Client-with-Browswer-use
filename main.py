import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any
from LLClient import LLMClient, get_response
import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # os.getenv("GROQ_API_KEY")
        self.api_key = "sk-xffqflnmcuojmizsgjgutppaejvplzckunbbffrimcjdmxcd"
        # self.api_key = os.getenv("GITHUB_API_KEY")


    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        server_params = StdioServerParameters(
            command=(
                shutil.which("npx")
                if self.config["command"] == "npx"
                else self.config["command"]
            ),
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            self.capabilities = await self.session.initialize()
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        supports_progress = self.capabilities and "progress" in self.capabilities

        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(
                            f"Tool '{tool.name}' will support progress tracking"
                        )
        # print([tools])
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities and "progress" in self.capabilities
                )

                if supports_progress:
                    logging.info(f"Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name, arguments, progress_token=f"{tool_name}_execution"
                    )
                else:
                    logging.info(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(
                            f"Warning during session cleanup for {self.name}: {e}"
                        )
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(
                            f"Note: Normal shutdown message for {self.name}: {e}"
                        )
                    except Exception as e:
                        logging.warning(
                            f"Warning during stdio cleanup for {self.name}: {e}"
                        )
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_gemini(self) -> dict:
        """Format tool as JSON structure for Gemini's toolConfig.functionDeclarations."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client


    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            await server.cleanup()

    async def process_llm_response(self, tool_call: dict) -> str:
  
        executed = False
        for server in self.servers:
            tools = await server.list_tools()
            if any(tool.name == tool_call["tool"] for tool in tools):
                try:
                    result = await server.execute_tool(
                        tool_call["tool"], tool_call["arguments"]
                    )
                    result_texts = [item.text for item in result.content if item.type == "text"]
                    executed = True
                    logging.info(f"Tool execution result: {result}")
                    break  
                except Exception as e:
                    error_msg = f"Error executing tool {tool_call['tool']}: {str(e)}"
                    logging.error(error_msg)
                    break
        if not executed:
            logging.info(f"No server found with tool: {tool_call['tool']}")
        return result_texts

    def buildContent(self, role, userInput)-> Dict:
        aContect = {
            "role": role,
            "parts": [
                {
                    "text":userInput
                }
            ]
        }
        return aContect

    def buildToolContent(self, role,tool_name, result)-> Dict:
        aContect = {
            "role": role,
            "parts": [
                {
                    "functionResponse": {
                        "name": tool_name,
                        "response": {
                            "result": result
                        }
                    }
                }

            ]
        }
        return aContect    

    def buildModelContent(self, functionCall)-> Dict:
        aContect = {
            "role": "model",
            "parts": functionCall
        }
        return aContect 

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return
            playLoad = {"contents":[],
                        "tools": {
                            "function_declarations":[]
                            }
                        }
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
            
            toolsList = [tool.format_for_gemini() for tool in all_tools]
            toolConfig = {"tools": {
                            "functionDeclarations":toolsList
                            }
                        }
            playLoad["tools"]["function_declarations"].extend(toolsList)

            # print("**************************************json.dumps(tool)*************************************************")
            # for tool in toolsList:
            #     print(json.dumps(toolConfig))
            # print("********************************************************************************")

            # playLoad.append(toolConfig)

            
            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break
                    print(user_input)
                    playLoad["contents"].append(self.buildContent("user",user_input))
                    

                    
                    is_continue = True
                    while is_continue:
                        llm_responses = self.llm_client.get_response(playLoad)
                        for response in llm_responses:
                            if isinstance(response, dict) and "functionCall" in response:
                                has_function_call = True
                                tool_call = {
                                    'tool': response['functionCall']['name'],
                                    'arguments': response['functionCall']['args']
                                }
                                
                                playLoad["contents"].append(self.buildModelContent(response))
                                result = await self.process_llm_response(tool_call)
                                playLoad["contents"].append(self.buildToolContent("user",response['functionCall']['name'],result))
                            else:
                                is_continue = False
                        else:
                            print(response)
                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    mcpServer_config = config.load_config("servers_config.json")
    mcpServers = [
        Server(name, srv_config)
        for name, srv_config in mcpServer_config["mcpServers"].items()
    ]
    llm_client = LLMClient()
    chat_session = ChatSession(mcpServers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())


