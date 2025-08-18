import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import aiohttp
from datetime import datetime

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    Prompt
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamableHTTPClient:
    """Custom HTTP client for streamable HTTP transport"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
    
    async def start(self):
        """Start the HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
    
    async def send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request"""
        if not self.session:
            raise RuntimeError("HTTP client not started")
        
        self._request_id += 1
        request_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method
        }
        
        if params:
            request_data["params"] = params
        
        async with self.session.post(
            self.base_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        ) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            result = await response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
            
            return result.get("result", {})
def create_http_streams(base_url: str):
    """Create read/write streams for HTTP transport"""
    http_client = StreamableHTTPClient(base_url)
    
    class WriteStream:
        def __init__(self, client):
            self.client = client
            self._started = False
        
        async def send(self, message):
            if not self._started:
                await self.client.start()
                self._started = True
            
            # Convert message to dict for HTTP transport
            if hasattr(message, 'model_dump'):
                message_dict = message.model_dump()
            elif hasattr(message, 'dict'):
                message_dict = message.dict()
            else:
                message_dict = dict(message)
            
            method = message_dict.get("method")
            params = message_dict.get("params", {})
            
            result = await self.client.send_request(method, params)
            
            # Return mock response
            return {
                "jsonrpc": "2.0",
                "id": message_dict.get("id"),
                "result": result
            }
        
        async def close(self):
            await self.client.close()
    
    class ReadStream:
        def __init__(self, write_stream):
            self.write_stream = write_stream
        
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            # This is a simplified implementation
            # In a real scenario, this would handle server-sent events
            raise StopAsyncIteration
    
    write_stream = WriteStream(http_client)
    read_stream = ReadStream(write_stream)
    
    return read_stream, write_stream
        
class LeaveManagerMCPClient:
    """MCP Client for LeaveManager using official ClientSession"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client: Optional[ClientSession] = None
        self.write_stream = None
    
    async def __aenter__(self):
        # Create HTTP streams
        read_stream, self.write_stream = create_http_streams(self.server_url)
        
        # Create ClientSession
        import sys

        self.client = ClientSession("http://localhost:8000/mcp", write_stream=sys.stdout)
        
        # Initialize the session
        await self.client.initialize()
        
        logger.info("✅ MCP Client initialized successfully")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.write_stream:
            await self.write_stream.close()
        logger.info("🔌 MCP Client disconnected")
    
    # Official MCP ClientSession methods
    
    async def list_tools(self) -> List[Tool]:
        """List available tools using official client method"""
        tools = await self.client.list_tools()
        logger.info(f"📋 Listed {len(tools.tools)} tools")
        return tools.tools
    
    async def list_resources(self) -> List[Resource]:
        """List available resources using official client method"""
        resources = await self.client.list_resources()
        logger.info(f"📚 Listed {len(resources.resources)} resources")
        return resources.resources
    
    async def list_prompts(self) -> List[Prompt]:
        """List available prompts using official client method"""
        prompts = await self.client.list_prompts()
        logger.info(f"💭 Listed {len(prompts.prompts)} prompts")
        return prompts.prompts
    
    async def call_tool(self, name: str, arguments: Dict[str, Any] = None):
        """Call a tool using official client method"""
        result = await self.client.call_tool(name, arguments or {})
        logger.info(f"🔧 Called tool: {name}")
        return result
    
    async def read_resource(self, uri: str):
        """Read a resource using official client method"""
        result = await self.client.read_resource(uri)
        logger.info(f"📖 Read resource: {uri}")
        return result
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None):
        """Get a prompt using official client method"""
        result = await self.client.get_prompt(name, arguments or {})
        logger.info(f"💬 Got prompt: {name}")
        return result
    
    # High-level convenience methods for LeaveManager
    
    async def get_leave_history(self, student_id: str) -> str:
        """Get leave history for a student"""
        result = await self.call_tool("get_leave_history", {"student_id": student_id})
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return "No leave history found"
    
    async def generate_poem(self, theme: str) -> str:
        """Generate a poem using AI"""
        result = await self.call_tool("poet", {"theme": theme})
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return "No poem generated"
    
    async def fetch_weather_batch(self, cities: List[str]) -> Dict[str, str]:
        """Fetch weather for multiple cities"""
        result = await self.call_tool("fetch_weather_batch", {"cities": cities})
        if result.content and len(result.content) > 0:
            try:
                response_text = result.content[0].text
                return json.loads(response_text) if isinstance(response_text, str) else response_text
            except json.JSONDecodeError:
                return {"error": "Failed to parse weather response"}
        return {}
    
    async def run_heavy_task(self, data: str) -> str:
        """Run a heavy computational task"""
        result = await self.call_tool("heavy_task", {"data": data})
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return "Heavy task failed"
    
    async def get_greeting(self, name: str) -> str:
        """Get personalized greeting"""
        result = await self.read_resource(f"greeting://{name}")
        if result.contents and len(result.contents) > 0:
            return result.contents[0].text
        return f"Hello, {name}!"
    
    async def get_energy_report(self) -> Dict[str, Any]:
        """Get energy and carbon usage report"""
        result = await self.read_resource("greenai://energy")
        if result.contents and len(result.contents) > 0:
            try:
                content = result.contents[0].text
                return json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                return {"error": "Failed to parse energy report"}
        return {}
    
    async def get_greenai_info(self) -> Dict[str, Any]:
        """Get Green AI practices and resource usage info"""
        result = await self.read_resource("greenai://info")
        if result.contents and len(result.contents) > 0:
            try:
                content = result.contents[0].text
                return json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                return {"error": "Failed to parse Green AI info"}
        return {}


async def demonstrate_mcp_client():
    """Demonstrate the MCP client using await client.list_tools() pattern"""
    print("🚀 LeaveManager MCP Client Demo")
    print("📡 Using official mcp.client.session.ClientSession")
    print("=" * 60)
    
    try:
        async with LeaveManagerMCPClient() as client:
            print("\n📋 Listing Available Tools:")
            print("-" * 30)
            tools = await client.list_tools()
            for tool in tools:
                print(f"  🔧 {tool.name}")
                print(f"     📝 {tool.description}")
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    properties = tool.inputSchema.get('properties', {})
                    if properties:
                        print(f"     📊 Parameters: {', '.join(properties.keys())}")
                print()
            
            print("\n📚 Listing Available Resources:")
            print("-" * 30)
            resources = await client.list_resources()
            for resource in resources:
                print(f"  📖 {resource.uri}")
                print(f"     📝 {resource.name}")
                if hasattr(resource, 'description') and resource.description:
                    print(f"     ℹ️  {resource.description}")
                print()
            
            print("\n💭 Listing Available Prompts:")
            print("-" * 30)
            prompts = await client.list_prompts()
            for prompt in prompts:
                print(f"  💬 {prompt.name}")
                print(f"     📝 {prompt.description}")
                print()
            
            print("\n🧪 Testing Tool Calls:")
            print("-" * 30)
            
            # Test leave history
            print("👨‍🎓 Testing Leave History:")
            try:
                history_s001 = await client.get_leave_history("S001")
                print(f"   S001: {history_s001}")
                
                history_unknown = await client.get_leave_history("S999")
                print(f"   S999: {history_unknown}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test greeting resource
            print("\n👋 Testing Greeting Resource:")
            try:
                greeting = await client.get_greeting("Alice")
                print(f"   {greeting}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test Green AI info
            print("\n🌱 Testing Green AI Info:")
            try:
                green_info = await client.get_greenai_info()
                if isinstance(green_info, dict):
                    practices = green_info.get('green_ai_practices', [])
                    print(f"   Practices: {practices}")
                    
                    usage = green_info.get('resource_usage', {})
                    print(f"   Memory: {usage.get('memory_rss_mb', 0)} MB")
                    print(f"   CPU: {usage.get('cpu_percent', 0)}%")
                    print(f"   Uptime: {usage.get('uptime_seconds', 0)} seconds")
                else:
                    print(f"   {green_info}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test energy report
            print("\n⚡ Testing Energy Report:")
            try:
                energy = await client.get_energy_report()
                if isinstance(energy, dict):
                    print(f"   Energy: {energy.get('estimated_energy_kwh', 0)} kWh")
                    print(f"   Carbon: {energy.get('estimated_carbon_kg', 0)} kg CO2")
                    print(f"   Platform: {energy.get('platform', 'Unknown')}")
                else:
                    print(f"   {energy}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test AI poem generation
            print("\n🎭 Testing AI Poem Generation:")
            try:
                poem = await client.generate_poem("green technology and sustainability")
                print(f"   Generated Poem:\n{poem}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test heavy task
            print("\n⚙️ Testing Heavy Task:")
            try:
                task_result = await client.run_heavy_task("climate data analysis")
                print(f"   Result: {task_result}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Test weather batch
            print("\n🌤️ Testing Weather Batch:")
            try:
                weather = await client.fetch_weather_batch(["Singapore", "London", "Tokyo"])
                for city, result in weather.items():
                    print(f"   {city}: {str(result)[:80]}{'...' if len(str(result)) > 80 else ''}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
    except Exception as e:
        print(f"\n❌ Client Error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


async def interactive_session():
    """Interactive session using the MCP client"""
    print("🔧 Interactive LeaveManager MCP Client")
    print("=" * 50)
    print("Commands:")
    print("  tools          - List available tools")
    print("  resources      - List available resources")
    print("  prompts        - List available prompts")
    print("  leave <id>     - Get leave history for student")
    print("  poem <theme>   - Generate a poem")
    print("  greeting <n>   - Get personalized greeting")
    print("  energy         - Get energy report")
    print("  info           - Get Green AI info")
    print("  heavy <data>   - Run heavy task")
    print("  weather <cities> - Get weather (comma-separated)")
    print("  quit           - Exit")
    print("=" * 50)
    
    try:
        async with LeaveManagerMCPClient() as client:
            print("✅ Connected to LeaveManager MCP Server")
            
            while True:
                try:
                    user_input = input("\n🔹 Enter command: ").strip()
                    if not user_input:
                        continue
                    
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    if command == "quit":
                        break
                    elif command == "tools":
                        tools = await client.list_tools()
                        print(f"\n📋 Available Tools ({len(tools)}):")
                        for tool in tools:
                            print(f"  🔧 {tool.name}: {tool.description}")
                    elif command == "resources":
                        resources = await client.list_resources()
                        print(f"\n📚 Available Resources ({len(resources)}):")
                        for resource in resources:
                            print(f"  📖 {resource.uri}: {resource.name}")
                    elif command == "prompts":
                        prompts = await client.list_prompts()
                        print(f"\n💭 Available Prompts ({len(prompts)}):")
                        for prompt in prompts:
                            print(f"  💬 {prompt.name}: {prompt.description}")
                    elif command == "leave":
                        student_id = arg or "S001"
                        result = await client.get_leave_history(student_id)
                        print(f"\n👨‍🎓 Leave History: {result}")
                    elif command == "poem":
                        theme = arg or "technology"
                        result = await client.generate_poem(theme)
                        print(f"\n🎭 Generated Poem:\n{result}")
                    elif command == "greeting":
                        name = arg or "User"
                        result = await client.get_greeting(name)
                        print(f"\n👋 Greeting: {result}")
                    elif command == "energy":
                        result = await client.get_energy_report()
                        print(f"\n⚡ Energy Report:")
                        print(json.dumps(result, indent=2))
                    elif command == "info":
                        result = await client.get_greenai_info()
                        print(f"\n🌱 Green AI Info:")
                        print(json.dumps(result, indent=2))
                    elif command == "heavy":
                        data = arg or "default processing task"
                        result = await client.run_heavy_task(data)
                        print(f"\n⚙️ Heavy Task Result: {result}")
                    elif command == "weather":
                        cities = [c.strip() for c in (arg or "Singapore,London").split(",")]
                        result = await client.fetch_weather_batch(cities)
                        print(f"\n🌤️ Weather Results:")
                        for city, weather_info in result.items():
                            print(f"  {city}: {weather_info}")
                    else:
                        print("❓ Unknown command. Type 'quit' to exit.")
                
                except KeyboardInterrupt:
                    print("\n👋 Exiting...")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
    
    print("👋 Goodbye!")


if __name__ == "__main__":
    print("LeaveManager MCP Client")
    print("Choose mode:")
    print("1. Demo mode (automated)")
    print("2. Interactive mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(interactive_session())
    else:
        asyncio.run(demonstrate_mcp_client())