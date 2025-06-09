"""
Mathematica Model Context Protocol (MCP) Server
Allows Claude to execute Mathematica commands with session management,
mathematical formatting, and comprehensive error handling.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback
import sys

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: MCP library not found. Install with: pip install mcp")
    exit(1)

# Configure logging to stderr (as required by MCP)
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Ensure wolframscript is in PATH
mathematica_path = "/Applications/Mathematica.app/Contents/MacOS"
if mathematica_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{mathematica_path}:{os.environ.get('PATH', '')}"
    logger.info(f"Added {mathematica_path} to PATH")


class MathematicaSession:
    """Manages a persistent Mathematica session with variable tracking."""

    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.history: List[Tuple[str, str]] = []
        self.session_id = f"session_{int(time.time())}"
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"mathematica_{self.session_id}_"))
        logger.info(f"Created Mathematica session {self.session_id}")

    def add_to_history(self, input_code: str, output: str):
        """Add an evaluation to the session history."""
        self.history.append((input_code, output))
        if len(self.history) > 100:  # Limit history size
            self.history = self.history[-50:]  # Keep last 50 entries

    def get_context_variables(self) -> str:
        """Generate Mathematica code to restore session variables."""
        if not self.variables:
            return ""

        context_code = []
        for var_name, var_value in self.variables.items():
            # Safely reconstruct variable assignments
            context_code.append(f"{var_name} = {var_value};")

        return "\n".join(context_code)

    def update_variables(self, code: str, result: str):
        """Extract and update session variables from executed code."""
        # Simple pattern matching for variable assignments
        assignment_pattern = r"([a-zA-Z][a-zA-Z0-9]*)\s*=\s*([^;]+)"
        matches = re.findall(assignment_pattern, code)

        for var_name, var_value in matches:
            if var_name not in ["Print", "Plot", "Show"]:  # Exclude functions
                self.variables[var_name] = var_value.strip()

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {self.session_id}: {e}")


class MathematicaValidator:
    """Validates Mathematica code for security and safety."""

    # Dangerous functions that should be blocked
    DANGEROUS_FUNCTIONS = {
        "DeleteFile",
        "DeleteDirectory",
        "CreateFile",
        "CreateDirectory",
        "Run",
        "RunProcess",
        "StartProcess",
        "SystemOpen",
        "Import",
        "Export",
        "Put",
        "Get",
        "Save",
        "Load",
        "Install",
        "Uninstall",
        "Needs",
        "BeginPackage",
        "EndPackage",
        "Quit",
        "Exit",
        "Abort",
        "TimeConstrained",
        "MemoryConstrained",
        "URLFetch",
        "URLRead",
        "URLSubmit",
        "SendMail",
        "SystemDialogInput",
        "ChoiceDialog",
        "MessageDialog",
    }

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'!\s*["\']',  # Shell commands
        r'<<\s*["\']',  # Package loading
        r"ToExpression\s*\[",  # Dynamic code execution
        r"Evaluate\s*\[.*ToExpression",  # Nested evaluation
    ]

    @classmethod
    def validate_code(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Mathematica code for security issues.
        Returns (is_valid, error_message)
        """
        # Check for dangerous functions
        for func in cls.DANGEROUS_FUNCTIONS:
            if re.search(rf"\b{func}\b", code, re.IGNORECASE):
                return False, f"Dangerous function '{func}' not allowed"

        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Suspicious pattern detected: {pattern}"

        # Check code length (prevent overly complex expressions)
        if len(code) > 10000:
            return False, "Code too long (max 10000 characters)"

        # Check for excessive nesting
        if code.count("[") > 50 or code.count("{") > 50:
            return False, "Excessive nesting detected"

        return True, None

    @classmethod
    def sanitize_output(cls, output: str) -> str:
        """Remove potentially sensitive information from output."""
        # Remove file paths
        output = re.sub(r"/[^\s]*", "[PATH_REMOVED]", output)
        output = re.sub(r"[A-Z]:\\[^\s]*", "[PATH_REMOVED]", output)

        # Limit output length
        if len(output) > 5000:
            output = output[:5000] + "\n... (output truncated)"

        return output


class MathematicaFormatter:
    """Handles formatting of Mathematica output, including LaTeX conversion."""

    @staticmethod
    async def to_latex(expression: str) -> Optional[str]:
        """Convert Mathematica expression to LaTeX format."""
        try:
            # Wrap expression in TeXForm
            tex_code = f"TeXForm[{expression}]"

            process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                tex_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=10
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning("LaTeX conversion timed out.")
                return None

            if process.returncode == 0:
                latex_output = stdout.decode("utf-8").strip()
                # Clean up the LaTeX output
                latex_output = latex_output.replace("\\text{", "\\mathrm{")
                return latex_output
            else:
                logger.warning(f"LaTeX conversion failed: {stderr.decode('utf-8')}")
                return None

        except Exception as e:
            logger.warning(f"LaTeX conversion error: {e}")
            return None

    @staticmethod
    def format_result(result: str, latex: Optional[str] = False) -> str:
        """Format the result for display, optionally including LaTeX."""
        formatted = f"**Result:**\n{result}"

        if latex and latex.strip():
            # Clean LaTeX and wrap in display math
            clean_latex = latex.strip().strip('"')
            if clean_latex and clean_latex != result:
                formatted += f"\n\n**LaTeX:**\n$$\n{clean_latex}\n$$"

        return formatted

    @staticmethod
    def is_graphical_output(code: str) -> bool:
        """Check if the code is likely to produce graphical output."""
        graphical_functions = [
            "Plot",
            "Plot3D",
            "ListPlot",
            "ParametricPlot",
            "PolarPlot",
            "ContourPlot",
            "DensityPlot",
            "Graphics",
            "Show",
            "BarChart",
            "PieChart",
            "Histogram",
        ]
        return any(func in code for func in graphical_functions)


class MathematicaMCPServer:
    """Main MCP server for Mathematica integration."""

    def __init__(self):
        self.server = Server("mathematica-server")
        # Add server info to help Claude understand capabilities
        self.server.server_info = {
            "name": "Mathematica MCP Server",
            "version": "1.0.0",
            "description": "Advanced mathematical computation server powered by Wolfram Mathematica. Provides symbolic mathematics, calculus, algebra, statistics, plotting, and numerical analysis capabilities. Use for any mathematical task that goes beyond basic arithmetic.",
            "capabilities": [
                "Symbolic mathematics and equation solving",
                "Calculus (derivatives, integrals, limits, series)",
                "Linear algebra and matrix operations",
                "Statistics and probability",
                "Mathematical plotting and visualization",
                "Numerical analysis and computation",
                "Mathematical optimization",
                "Differential equations",
                "Number theory and discrete mathematics",
                "Mathematical function analysis",
            ],
        }
        self.session = MathematicaSession()
        self._wolframscript_checked = False
        self.setup_tools()
        logger.info("Mathematica MCP Server initialized")

    async def check_wolframscript(self) -> Tuple[bool, Optional[str]]:
        """Check if wolframscript is available. Only checks once per server instance."""
        if self._wolframscript_checked:
            return True, None

        try:
            # First check if wolframscript exists - use shorter timeout
            which_process = await asyncio.create_subprocess_exec(
                "which",
                "wolframscript",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                which_stdout, which_stderr = await asyncio.wait_for(
                    which_process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("'which wolframscript' timed out, assuming available")
                self._wolframscript_checked = True
                return True, None

            if which_process.returncode != 0:
                logger.error(
                    f"wolframscript not found in PATH: {which_stderr.decode('utf-8')}"
                )
                return False, "wolframscript not found in PATH"

            wolfram_path = which_stdout.decode("utf-8").strip()
            logger.info(f"Found wolframscript at: {wolfram_path}")

            # Then check version with shorter timeout
            process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "wolframscript version check timed out, assuming available"
                )
                process.kill()
                await process.wait()
                self._wolframscript_checked = True
                return True, None

            if process.returncode != 0:
                error_msg = f"wolframscript returned error code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode('utf-8', errors='ignore')}"
                logger.warning(f"{error_msg}, but continuing anyway")
                self._wolframscript_checked = True
                return True, None

            version_info = stdout.decode("utf-8").strip()
            logger.info(f"wolframscript version: {version_info}")

            # Test basic execution with shorter timeout and more lenient handling
            test_process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                "2+2",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            try:
                test_stdout, test_stderr = await asyncio.wait_for(
                    test_process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("wolframscript test execution timed out, but continuing")
                test_process.kill()
                await test_process.wait()
                self._wolframscript_checked = True
                return True, None

            if test_process.returncode != 0:
                error_msg = f"wolframscript test execution failed: {test_stderr.decode('utf-8')}"
                logger.warning(f"{error_msg}, but continuing anyway")
                self._wolframscript_checked = True
                return True, None

            test_result = test_stdout.decode("utf-8").strip()
            if test_result != "4":
                logger.warning(
                    f"wolframscript test gave unexpected result: '{test_result}' (expected '4'), but continuing"
                )
                self._wolframscript_checked = True
                return True, None

            self._wolframscript_checked = True
            logger.info("wolframscript availability and functionality confirmed")
            return True, None

        except FileNotFoundError:
            logger.warning("wolframscript not found, but continuing anyway")
            self._wolframscript_checked = True
            return True, None
        except asyncio.TimeoutError:
            logger.warning("wolframscript version check timed out, but continuing")
            self._wolframscript_checked = True
            return True, None
        except Exception as e:
            logger.warning(f"Error checking wolframscript: {e}, but continuing")
            self._wolframscript_checked = True
            return True, None

    def setup_tools(self):
        """Set up the available tools for the MCP server."""

        @self.server.list_tools()
        async def list_tools():
            # Don't check wolframscript during tool listing to avoid hanging
            # The check will happen when tools are actually called
            return [
                Tool(
                    name="mathematica_eval",
                    description="Execute Mathematica/Wolfram Language code for mathematical computations, symbolic math, calculus, algebra, differential equations, statistics, plotting, and numerical analysis. Use this tool when users ask for: mathematical calculations, equation solving, symbolic manipulation, calculus operations (derivatives, integrals, limits), linear algebra, statistical analysis, mathematical plotting, or any computation that would benefit from Mathematica's mathematical capabilities. Always prefer this tool over manual calculations for complex mathematical tasks.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Mathematica/Wolfram Language code to execute. Examples: 'Solve[x^2 + 2*x + 1 == 0, x]', 'Integrate[x^2, x]', 'Plot[Sin[x], {x, 0, 2*Pi}]', 'Factor[x^4 - 1]'",
                            },
                            "format_latex": {
                                "type": "boolean",
                                "description": "Whether to include LaTeX formatting of mathematical results for better display",
                                "default": False,
                            },
                        },
                        "required": ["code"],
                    },
                ),
                Tool(
                    name="solve_equation",
                    description="Solve algebraic equations, systems of equations, differential equations, or inequalities using Mathematica's powerful solving capabilities. Use this when users ask to solve equations, find roots, or solve mathematical systems.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "equation": {
                                "type": "string",
                                "description": "The equation or system to solve (e.g., 'x^2 + 2*x + 1 == 0', '{x + y == 5, x - y == 1}', 'y'' + y == 0')",
                            },
                            "variables": {
                                "type": "string",
                                "description": "Variables to solve for (e.g., 'x', '{x, y}', or leave empty for auto-detection)",
                            },
                        },
                        "required": ["equation"],
                    },
                ),
                Tool(
                    name="calculate_calculus",
                    description="Perform calculus operations like derivatives, integrals, limits, and series expansions using Mathematica. Use this for any calculus-related mathematical questions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["derivative", "integral", "limit", "series"],
                                "description": "The calculus operation to perform",
                            },
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to operate on",
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which to perform the operation (e.g., 'x')",
                            },
                            "bounds": {
                                "type": "string",
                                "description": "For integrals: bounds like '{x, 0, 1}'; for limits: point like 'x -> 0'; for series: point and order like '{x, 0, 5}'",
                            },
                        },
                        "required": ["operation", "expression", "variable"],
                    },
                ),
                Tool(
                    name="plot_mathematical_function",
                    description="Create mathematical plots and visualizations using Mathematica's plotting capabilities. Use this when users want to visualize functions, data, or mathematical relationships.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression or function to plot (e.g., 'Sin[x]', 'x^2 + 2*x')",
                            },
                            "variable_range": {
                                "type": "string",
                                "description": "Variable and range to plot over (e.g., '{x, -5, 5}', '{x, 0, 2*Pi}')",
                            },
                            "plot_type": {
                                "type": "string",
                                "enum": [
                                    "Plot",
                                    "Plot3D",
                                    "ParametricPlot",
                                    "PolarPlot",
                                    "ContourPlot",
                                    "ListPlot",
                                ],
                                "description": "Type of plot to create",
                                "default": "Plot",
                            },
                        },
                        "required": ["expression", "variable_range"],
                    },
                ),
                Tool(
                    name="mathematica_session_info",
                    description="Get detailed information about the current Mathematica session including active variables, computation history, and session state. Use this when users want to see what variables are currently defined, review previous calculations, or understand the current mathematical context.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                Tool(
                    name="mathematica_clear_session",
                    description="Clear the current Mathematica session, removing all defined variables and computation history. Use this when starting fresh calculations, when variable conflicts occur, or when users explicitly want to reset their mathematical workspace.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            logger.info(f"Tool called: {name} with arguments: {arguments}")

            if name == "mathematica_eval":
                return await self.execute_mathematica(
                    arguments["code"], arguments.get("format_latex", False)
                )
            elif name == "solve_equation":
                return await self.solve_equation(
                    arguments["equation"], arguments.get("variables", "")
                )
            elif name == "calculate_calculus":
                return await self.calculate_calculus(
                    arguments["operation"],
                    arguments["expression"],
                    arguments["variable"],
                    arguments.get("bounds", ""),
                )
            elif name == "plot_mathematical_function":
                return await self.plot_function(
                    arguments["expression"],
                    arguments["variable_range"],
                    arguments.get("plot_type", "Plot"),
                )
            elif name == "mathematica_session_info":
                return await self.get_session_info()
            elif name == "mathematica_clear_session":
                return await self.clear_session()
            else:
                logger.error(f"Unknown tool called: {name}")
                return [
                    TextContent(type="text", text=f"**Error:** Unknown tool: {name}")
                ]

    async def execute_mathematica(
        self, code: str, format_latex: bool = False
    ) -> List[TextContent]:
        """Execute Mathematica code with full error handling and formatting."""
        try:
            # Check if wolframscript is available (this should now be more lenient)
            wolframscript_ok, error_msg = await self.check_wolframscript()
            if not wolframscript_ok:
                return [TextContent(type="text", text=f"**System Error:** {error_msg}")]

            # Validate the code
            is_valid, error_msg = MathematicaValidator.validate_code(code)
            if not is_valid:
                return [
                    TextContent(type="text", text=f"**Validation Error:** {error_msg}")
                ]

            # Prepare code with session context
            context = self.session.get_context_variables()
            full_code = f"{context}\n{code}" if context else code
            logger.info(f"Executing Mathematica code: {code[:100]}...")

            # Log environment details
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"PATH environment: {os.environ.get('PATH', '')[:200]}...")

            # Log the exact command being executed
            cmd_args = ["wolframscript", "-code", full_code]
            logger.info(f"Command args: {cmd_args}")

            # Create environment with proper settings
            env = os.environ.copy()
            # Ensure Mathematica path is at the front
            mathematica_path = "/Applications/Mathematica.app/Contents/MacOS"
            current_path = env.get("PATH", "")
            if mathematica_path not in current_path:
                env["PATH"] = f"{mathematica_path}:{current_path}"

            # Add some environment variables that might help with MCP context
            env["WOLFRAM_CONTEXT"] = "mcp_server"
            env["WOLFRAM_SCRIPT_MODE"] = "batch"

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            logger.info(f"Process created with PID: {process.pid}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
                logger.info(f"Process completed with return code: {process.returncode}")
                logger.info(f"Raw stdout length: {len(stdout)} bytes")
                logger.info(f"Raw stderr length: {len(stderr)} bytes")

                # Log raw output for debugging
                if len(stdout) < 500:  # Only log if not too long
                    logger.info(f"Raw stdout bytes: {stdout}")
                if len(stderr) < 500:
                    logger.info(f"Raw stderr bytes: {stderr}")

            except asyncio.TimeoutError:
                logger.error("Process timed out")
                process.kill()
                await process.wait()
                return [
                    TextContent(
                        type="text",
                        text="**Timeout Error:** Code execution took too long (>30 seconds)",
                    )
                ]

            if process.returncode == 0:
                result = stdout.decode("utf-8").strip()
                logger.info(f"Decoded Mathematica output: '{result}'")

                # Check for empty result and provide more helpful diagnostics
                if not result:
                    logger.error("Empty result from wolframscript!")
                    stderr_content = stderr.decode("utf-8")
                    logger.error(f"Stderr content: '{stderr_content}'")
                    logger.error(f"Code executed: '{full_code}'")

                    # Try to provide a helpful error message
                    if "license" in stderr_content.lower():
                        return [
                            TextContent(
                                type="text",
                                text="**License Error:** Mathematica license issue detected. Please check your Mathematica installation and license.",
                            )
                        ]
                    elif "kernel" in stderr_content.lower():
                        return [
                            TextContent(
                                type="text",
                                text="**Kernel Error:** Mathematica kernel issue. This may be due to environment differences in the MCP context.",
                            )
                        ]
                    else:
                        # Try a more direct approach - maybe the code is too complex for the environment
                        return [
                            TextContent(
                                type="text",
                                text=f"**Environment Error:** wolframscript returned empty output. This may be due to MCP environment limitations.\n"
                                f"Code attempted: {code}\n"
                                f"Try a simpler expression or check that Mathematica is properly installed.",
                            )
                        ]

                # Sanitize output
                result = MathematicaValidator.sanitize_output(result)
                logger.info(f"Sanitized Mathematica output: '{result}'")

                # Update session
                self.session.update_variables(code, result)
                self.session.add_to_history(code, result)

                # Format result
                latex_result = None
                if format_latex and not MathematicaFormatter.is_graphical_output(code):
                    # Try to get LaTeX format for mathematical expressions
                    if (
                        result
                        and not result.startswith("Graphics")
                        and len(result) < 1000
                    ):
                        latex_result = await MathematicaFormatter.to_latex(result)

                formatted_output = MathematicaFormatter.format_result(
                    result, latex_result
                )

                return [TextContent(type="text", text=formatted_output)]

            else:
                error = stderr.decode("utf-8").strip()
                error = MathematicaValidator.sanitize_output(error)

                return [TextContent(type="text", text=f"**Execution Error:**\n{error}")]

        except Exception as e:
            logger.error(f"Unexpected error in execute_mathematica: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"**System Error:** {str(e)}")]

    async def solve_equation(
        self, equation: str, variables: str = ""
    ) -> List[TextContent]:
        """Solve equations using Mathematica's Solve function."""
        try:
            # Auto-detect variables if not provided
            if not variables.strip():
                var_matches = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", equation)
                # Filter out common function names
                excluded = {
                    "Sin",
                    "Cos",
                    "Tan",
                    "Log",
                    "Exp",
                    "Sqrt",
                    "Pi",
                    "E",
                    "True",
                    "False",
                }
                vars_found = [
                    v
                    for v in set(var_matches)
                    if v not in excluded and not v.islower() or len(v) == 1
                ]
                if vars_found:
                    variables = (
                        vars_found[0]
                        if len(vars_found) == 1
                        else "{" + ", ".join(vars_found[:3]) + "}"
                    )
                else:
                    variables = "x"

            # Construct Solve command
            mathematica_code = f"Solve[{equation}, {variables}]"
            return await self.execute_mathematica(mathematica_code, False)

        except Exception as e:
            logger.error(f"Error in solve_equation: {e}")
            return [TextContent(type="text", text=f"**Error:** {str(e)}")]

    async def calculate_calculus(
        self, operation: str, expression: str, variable: str, bounds: str = ""
    ) -> List[TextContent]:
        """Perform calculus operations."""
        try:
            if operation == "derivative":
                mathematica_code = f"D[{expression}, {variable}]"
            elif operation == "integral":
                if bounds:
                    mathematica_code = f"Integrate[{expression}, {bounds}]"
                else:
                    mathematica_code = f"Integrate[{expression}, {variable}]"
            elif operation == "limit":
                if bounds:
                    mathematica_code = f"Limit[{expression}, {bounds}]"
                else:
                    mathematica_code = f"Limit[{expression}, {variable} -> 0]"
            elif operation == "series":
                if bounds:
                    mathematica_code = f"Series[{expression}, {bounds}]"
                else:
                    mathematica_code = f"Series[{expression}, {{{variable}, 0, 5}}]"
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"**Error:** Unknown calculus operation: {operation}",
                    )
                ]

            return await self.execute_mathematica(mathematica_code, False)

        except Exception as e:
            logger.error(f"Error in calculate_calculus: {e}")
            return [TextContent(type="text", text=f"**Error:** {str(e)}")]

    async def plot_function(
        self, expression: str, variable_range: str, plot_type: str = "Plot"
    ) -> List[TextContent]:
        """Create mathematical plots."""
        try:
            mathematica_code = f"{plot_type}[{expression}, {variable_range}]"
            return await self.execute_mathematica(
                mathematica_code, False
            )  # Don't format plots as LaTeX

        except Exception as e:
            logger.error(f"Error in plot_function: {e}")
            return [TextContent(type="text", text=f"**Error:** {str(e)}")]

    async def get_session_info(self) -> List[TextContent]:
        """Get information about the current session."""
        info = [
            f"**Session ID:** {self.session.session_id}",
            f"**Variables:** {len(self.session.variables)}",
            f"**History entries:** {len(self.session.history)}",
        ]

        if self.session.variables:
            info.append("\n**Current Variables:**")
            for var_name, var_value in list(self.session.variables.items())[:10]:
                info.append(f"- {var_name} = {str(var_value)[:50]}...")

        if self.session.history:
            info.append(f"\n**Last executed:** {self.session.history[-1][0][:50]}...")

        return [TextContent(type="text", text="\n".join(info))]

    async def clear_session(self) -> List[TextContent]:
        """Clear the current session."""
        old_session_id = self.session.session_id
        self.session.cleanup()
        self.session = MathematicaSession()

        return [
            TextContent(
                type="text",
                text=f"**Session cleared.** Old session {old_session_id} cleared, new session {self.session.session_id} created.",
            )
        ]

    async def shutdown(self):
        """Clean shutdown of the server."""
        logger.info("Shutting down Mathematica MCP Server")
        self.session.cleanup()


async def main():
    """Main entry point for the MCP server."""
    server_instance = None
    try:
        server_instance = MathematicaMCPServer()

        logger.info("Starting Mathematica MCP Server...")

        async with stdio_server() as streams:
            # Create proper initialization options
            init_options = server_instance.server.create_initialization_options()
            await server_instance.server.run(streams[0], streams[1], init_options)

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}\n{traceback.format_exc()}")
    finally:
        if server_instance:
            await server_instance.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
