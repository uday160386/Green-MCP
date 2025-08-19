# Green-MCP

Green-MCP is an AI-powered Model Context Protocol (MCP) server and client project designed to facilitate generative AI workflows and context management. It provides a Python-based server and client, with optional UI and data handling components.

## Features
- MCP server implementation (`server/green-mcp-server.py`)
- Python client for interacting with the server (`client/green-mcp-client.py`)
- Streamlit UI for client interaction (`client/green-mcp-streamlit.py`)
- Docker support for easy deployment
- Example data and documentation

## Project Structure
```
Green-MCP/
├── Dockerfile
├── main.py
├── mcp-server-openrpc.json
├── pyproject.toml
├── requirements.txt
├── README.md
├── client/
│   ├── green-mcp-client.py
│   └── green-mcp-streamlit.py
├── client-ui/
├── data/
│   └── Generative_AI_1705404080.pdf
├── server/
│   └── green-mcp-server.py
```

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional)
- pip

### Installation
1. Clone the repository:
	```bash
	git clone https://github.com/uday160386/Green-MCP.git
	cd Green-MCP
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

### Running the Server
```bash
python server/green-mcp-server.py
```

### Running the Client
```bash
python client/green-mcp-client.py
```

### Using the Streamlit UI
```bash
streamlit run client/green-mcp-streamlit.py
```

### Docker Usage
Build and run the container:
```bash
docker build -t green-mcp .
docker run -p 8000:8000 green-mcp
```

## Documentation
- See `mcp-server-openrpc.json` for the OpenRPC specification.
- Example data in `data/Generative_AI_1705404080.pdf`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
