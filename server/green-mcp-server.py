from mcp.server.fastmcp import FastMCP
from typing import List
import psutil
import time
import os
import logging
import asyncio
from functools import lru_cache
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from dotenv import load_dotenv


# In-memory mock database with 20 leave days to start
student_leaves = {
    "S001": {"balance": 18, "history": ["2024-12-25", "2025-01-01"]},
    "S002": {"balance": 20, "history": []}
}

# Create MCP server
mcp = FastMCP("LeaveManager", stateless_http=True,  host="0.0.0.0", port=8000)

# Setup logging for resource monitoring
logging.basicConfig(filename="greenai.log", level=logging.INFO)

load_dotenv()


# Green AI: Adaptive model selection
def get_model(task: str):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
    if task == "poetry":
        # Use a smaller model for non-critical tasks
        return AnthropicModel('claude-3-haiku-latest', provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY))
    # Default to larger model for critical tasks
    return AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY))

def get_agent(task: str):
    return Agent(get_model(task))


# Green AI: Adaptive model selection for poetry
@mcp.tool()
async def poet(theme: str) -> str:
    """Poem generator (adaptive model selection)"""
    agent = get_agent("poetry")
    r = await agent.run(f'write a poem about {theme}')
    return r.output

# Green AI: async weather fetch with lazy import and request throttling
import time as _time
_last_weather_call = {}
WEATHER_THROTTLE_SECONDS = int(os.getenv("WEATHER_THROTTLE_SECONDS", "10"))


# Green AI: Batch weather processing
@mcp.tool()
async def fetch_weather_batch(cities: List[str]) -> dict:
    """Batch fetch weather for multiple cities (async, lazy import, throttled)"""
    import httpx
    results = {}
    now = _time.time()
    async with httpx.AsyncClient() as client:
        for city in cities:
            last_call = _last_weather_call.get(city)
            if last_call and now - last_call < WEATHER_THROTTLE_SECONDS:
                results[city] = f"Weather API call for {city} throttled. Try again in {int(WEATHER_THROTTLE_SECONDS - (now - last_call))} seconds."
                continue
            try:
                response = await client.get(f"https://api.weather.com/{city}")
                _last_weather_call[city] = _time.time()
                results[city] = response.text
            except Exception as e:
                import logging
                logging.error(f"Weather API error: {e}")
                results[city] = f"Weather API error: {e}"
    return results

# Green AI: Energy/Carbon reporting
@mcp.resource("greenai://energy")
def energy_report():
    """Estimate energy and carbon usage based on server metrics."""
    import platform
    import psutil
    import os
    # Simple estimation: energy = uptime * avg_cpu_percent * factor
    process = psutil.Process(os.getpid())
    uptime = time.time() - process.create_time()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    # Assume 50W server, 0.5 kg CO2/kWh
    energy_kwh = (uptime * cpu_percent / 100 * 50) / 3600000
    carbon_kg = energy_kwh * 0.5
    return {
        "uptime_seconds": int(uptime),
        "avg_cpu_percent": cpu_percent,
        "estimated_energy_kwh": round(energy_kwh, 4),
        "estimated_carbon_kg": round(carbon_kg, 4),
        "platform": platform.platform()
    }

# Green AI: Environment-aware scheduling (example: restrict heavy tasks to off-peak hours)
def is_off_peak():
    import datetime
    hour = datetime.datetime.now().hour
    # Assume off-peak is 10pm-6am
    return hour >= 22 or hour < 6

@mcp.tool()
async def heavy_task(data: str) -> str:
    """Run a heavy task only during off-peak hours."""
    if not is_off_peak():
        return "Heavy tasks are restricted to off-peak hours (10pm-6am) for energy efficiency."
    # ...perform heavy computation...
    await asyncio.sleep(2)  # Simulate heavy work
    return f"Heavy task completed for: {data}"


# Resource: Leave history
@mcp.tool()
def get_leave_history(student_id: str) -> str:
    """Get leave history for the student"""
    data = student_leaves.get(student_id)
    if data:
        history = ', '.join(data['history']) if data['history'] else "No leaves taken."
        return f"Leave history for {student_id}: {history}"
    return "Student ID not found."



# Green AI: async PDF loading and caching with periodic cleanup
import functools
pdf_cache = {}
PDF_CACHE_SIZE = int(os.getenv("PDF_CACHE_SIZE", "1"))
PDF_CACHE_TIMEOUT = int(os.getenv("PDF_CACHE_TIMEOUT", "600"))  # seconds
pdf_cache_times = {}

async def cleanup_pdf_cache():
    now = _time.time()
    expired = [k for k, t in pdf_cache_times.items() if now - t > PDF_CACHE_TIMEOUT]
    for k in expired:
        pdf_cache.pop(k, None)
        pdf_cache_times.pop(k, None)

@mcp.resource("pdf://Generative_AI")
async def get_pdf() -> bytes:
    """Return the contents of the Generative_AI_1705404080.pdf file as bytes (async, cached, periodic cleanup)."""
    await cleanup_pdf_cache()
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "Generative_AI_1705404080.pdf")
    if pdf_path in pdf_cache:
        return pdf_cache[pdf_path]
    loop = asyncio.get_event_loop()
    with open(pdf_path, "rb") as f:
        data = await loop.run_in_executor(None, f.read)
    if len(pdf_cache) >= PDF_CACHE_SIZE:
        oldest = min(pdf_cache_times, key=pdf_cache_times.get)
        pdf_cache.pop(oldest, None)
        pdf_cache_times.pop(oldest, None)
    pdf_cache[pdf_path] = data
    pdf_cache_times[pdf_path] = _time.time()
    return data
    
@mcp.prompt()
def extract_text_from_pdf(resource: str) -> list[str]:
    """Extracts text from a PDF resource."""
    # Placeholder: Replace with actual PDF extraction logic
    return [f"Extracted text from {resource}:  Placeholder text."]

# Resource: Greeting
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}! How can I assist you with leave management today?"


@mcp.resource("greenai://info")
def greenai_info():
    """Report on green AI practices and resource usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    uptime = time.time() - process.create_time()
    return {
        "green_ai_practices": [
            "Efficient model selection",
            "Caching and batching where possible",
            "Responsible data handling",
            "Resource monitoring"
        ],
        "resource_usage": {
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "cpu_percent": cpu_percent,
            "uptime_seconds": int(uptime)
        }
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    