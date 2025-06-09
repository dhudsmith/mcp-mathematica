# Mathematica MCP Server

This project provides an MCP (Model Context Protocol) server for Mathematica, enabling safe, session-based Mathematica code execution for use with Claude Desktop or other MCP-compatible clients.

---

## Features

- **Session Management:** Persistent Mathematica sessions with variable tracking and history.
- **Secure Execution:** Code validation to block dangerous Mathematica functions and suspicious patterns.
- **MCP Integration:** Compatible with Claude's MCP tool interface.
- **OutputForm Only:** All results are returned in Mathematica's OutputForm (not TeXForm); Claude handles LaTeX formatting if needed.
- **Diagnostics Tool:** Includes a `test_wolframscript` tool for troubleshooting server/Mathematica connectivity.

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
   Ensure Mathematica is installed and `wolframscript` is available in your system PATH. The server will attempt to add `/Applications/Mathematica.app/Contents/MacOS` to your PATH automatically on macOS.

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

---

## Integration with Claude Desktop

To use this server with Claude Desktop's MCP tool interface:

1. **Locate your Claude Desktop config file:**  
   Usually at:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

2. **Add or update the Mathematica MCP server entry as follows:**
   ```json
   {
     "mcpServers": {
       "mathematica": {
         "command": "/path/to/your/conda/env/bin/python",
         "args": ["/path/to/your/wolfram_mcp/mcp_server.py"],
         "env": {
           "MATHEMATICA_TOOL_PREFERENCE": "high",
           "MATHEMATICAL_COMPUTATION_MODE": "advanced",
           "PATH": "/path/to/Mathematica.app/Contents/MacOS:/path/to/your/conda/env/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
           "HOME": "/your/username",
           "USER": "your_username",
           "WOLFRAM_KERNEL": "/path/to/Mathematica.app/Contents/MacOS/MathKernel",
           "MATHEMATICA_HOME": "/path/to/Mathematica.app/Contents",
           "WOLFRAM_SCRIPT_MODE": "batch",
           "PYTHONUNBUFFERED": "1",
           "WOLFRAM_CONTEXT": "mcp_server"
         }
       }
     },
     "globalSettings": {
       "toolUsage": {
         "mathematicalQueries": {
           "preferredTools": ["mathematica"],
           "triggerKeywords": [
             "solve", "integrate", "derivative", "differential", "equation",
             "calculus", "algebra", "plot", "graph", "mathematical", "computation",
             "symbolic", "factor", "expand", "simplify", "matrix", "determinant",
             "eigenvalue", "statistics", "probability", "optimization", "limit",
             "series", "taylor", "fourier", "laplace", "transform"
           ]
         }
       }
     }
   }
   ```
   **Note:** All local paths (such as the Python executable, MCP server script, Mathematica installation, and user-specific directories) must be updated to match your own system. Replace `/path/to/your/conda/env`, `/path/to/your/wolfram_mcp`, `/path/to/Mathematica.app`, `/your/username`, and `your_username` with the appropriate values for your setup.

3. **Restart Claude Desktop** to pick up the new MCP server configuration.

Claude will now use this Mathematica MCP server for mathematical queries and code execution, with the specified environment and tool preferences.

---

## Available MCP Tools

- **mathematica_eval**:  
  Execute Mathematica/Wolfram Language code for mathematical computations, symbolic math, calculus, algebra, differential equations, statistics, plotting, and numerical analysis. Use this for any advanced mathematical task.

- **test_wolframscript**:  
  Diagnostic tool to check if `wolframscript` is working correctly. Use this if you encounter issues with code execution or server startup.

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
- For custom Mathematica paths, edit the environment variables in `env.yaml` or ensure your PATH is set correctly.
- All output is returned in OutputForm (plain text); Claude will handle any further formatting or LaTeX conversion.
- Use the `test_wolframscript` tool for troubleshooting if you encounter issues with Mathematica connectivity.

---
