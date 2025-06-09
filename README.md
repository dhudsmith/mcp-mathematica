# Mathematica MCP Server

This project provides an MCP (Model Context Protocol) server for Mathematica, enabling safe, session-based Mathematica code execution for use with Claude or other MCP-compatible clients.

---

## Features

- **Session Management:** Persistent Mathematica sessions with variable tracking and history.
- **Secure Execution:** Code validation to block dangerous Mathematica functions.
- **MCP Integration:** Compatible with Claude's MCP tool interface.

---

## Environment Setup

1. **Install Conda**  
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) is required.

2. **Create the Environment**  
   In the project directory, run:
   ```sh
   conda env create -f env.yaml
   ```

3. **Activate the Environment**  
   ```sh
   conda activate mathematica-mcp
   ```

4. **Install Mathematica**  
   Ensure Mathematica is installed and `wolframscript` is available in your system PATH.

---

## Usage

### Running the MCP Server

Start the server manually:
```sh
python mcp_server.py
```

Or, if using the conda environment:
```sh
conda run -n mathematica-mcp python mcp_server.py
```

### Integration with Claude MCP

Configure Claude to use this server by adding the following to your Claude config (already present in your setup):

```json
{
  "mcpServers": {
    "mathematica": {
      "command": "conda",
      "args": ["run", "-n", "mathematica-mcp", "python", "/Users/dane2/code/wolfram_mcp/mcp_server.py"]
    }
  }
}
```
See [`/Users/dane2/Library/Application Support/Claude/claude_desktop_config.json`](../Library/Application%20Support/Claude/claude_desktop_config.json).

---

## Testing

A comprehensive test script is provided to verify server functionality.

### Run All Tests

```sh
python mcp_server_test.py ./mcp_server.py
```

The test script checks:
- Prerequisites (Mathematica, wolframscript, Python dependencies)
- Server import and class presence
- Code validation (security)
- Basic Mathematica execution
- Session management
- Server startup

---

## File Overview

- [`mcp_server.py`](mcp_server.py): Main MCP server implementation.
- [`mcp_server_test.py`](mcp_server_test.py): Automated test suite.
- [`env.yaml`](env.yaml): Conda environment specification.

---

## Notes

- The server requires Mathematica and `wolframscript` to be installed and accessible.
- The MCP Python package is installed via pip in the conda environment.
- For custom Mathematica paths, edit the environment variables in `env.yaml`.

---
