"""
Mathematica Model Context Protocol (MCP) Server - Fixed Version
Includes fixes for empty output issues in Claude Desktop environment
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
                "-print",
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
    def format_result(result: str, latex: Optional[str] = None) -> str:
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
            "version": "1.0.1",
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
            # Log environment for debugging
            logger.info(f"Current environment variables: {dict(os.environ)}")
            logger.info(f"Current working directory: {os.getcwd()}")

            # Test basic execution with explicit print
            test_process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                'Print["MCP_TEST_OK"]',
                "-print",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                test_stdout, test_stderr = await asyncio.wait_for(
                    test_process.communicate(), timeout=10
                )
            except asyncio.TimeoutError:
                logger.error("wolframscript test timed out")
                test_process.kill()
                await test_process.wait()
                return False, "wolframscript test timed out"

            stdout_str = test_stdout.decode("utf-8").strip()
            stderr_str = test_stderr.decode("utf-8").strip()

            logger.info(f"Test stdout: '{stdout_str}'")
            logger.info(f"Test stderr: '{stderr_str}'")
            logger.info(f"Test return code: {test_process.returncode}")

            if test_process.returncode != 0:
                return (
                    False,
                    f"wolframscript test failed with code {test_process.returncode}",
                )

            if "MCP_TEST_OK" not in stdout_str:
                logger.warning(f"Expected 'MCP_TEST_OK' in output, got: '{stdout_str}'")
                # Don't fail, just warn

            self._wolframscript_checked = True
            logger.info("wolframscript check passed")
            return True, None

        except FileNotFoundError:
            return False, "wolframscript not found in PATH"
        except Exception as e:
            logger.error(f"Error checking wolframscript: {e}")
            return False, f"Error checking wolframscript: {str(e)}"

    def setup_tools(self):
        """Set up the available tools for the MCP server."""

        @self.server.list_tools()
        async def list_tools():
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
                                "default": True,
                            },
                        },
                        "required": ["code"],
                    },
                ),
                Tool(
                    name="test_wolframscript",
                    description="Test if wolframscript is working correctly. Use this for debugging MCP connection issues.",
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
                    arguments["code"], arguments.get("format_latex", True)
                )
            elif name == "test_wolframscript":
                return await self.test_wolframscript_direct()
            else:
                logger.error(f"Unknown tool called: {name}")
                return [
                    TextContent(type="text", text=f"**Error:** Unknown tool: {name}")
                ]

    async def test_wolframscript_direct(self) -> List[TextContent]:
        """Direct test of wolframscript without any wrapping."""
        try:
            # Test 1: Basic execution
            process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                "2+2",
                "-print",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            result = f"Test 1 - Basic: stdout='{stdout.decode()}', stderr='{stderr.decode()}', code={process.returncode}\n"

            # Test 2: With explicit Print
            process2 = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                "Print[4]",
                "-print",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, stderr2 = await process2.communicate()

            result += f"Test 2 - Print: stdout='{stdout2.decode()}', stderr='{stderr2.decode()}', code={process2.returncode}\n"

            # Test 3: Using MathKernel directly
            kernel_path = "/Applications/Mathematica.app/Contents/MacOS/MathKernel"
            if os.path.exists(kernel_path):
                process3 = await asyncio.create_subprocess_exec(
                    kernel_path,
                    "-noprompt",
                    "-run",
                    "Print[6]; Exit[]",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout3, stderr3 = await process3.communicate()
                result += f"Test 3 - MathKernel: stdout='{stdout3.decode()}', stderr='{stderr3.decode()}', code={process3.returncode}\n"
            else:
                result += f"Test 3 - MathKernel: Not found at {kernel_path}\n"

            # Test 4: Environment info
            result += f"\nEnvironment:\n"
            result += f"PATH: {os.environ.get('PATH', 'NOT SET')}\n"
            result += f"HOME: {os.environ.get('HOME', 'NOT SET')}\n"
            result += f"CWD: {os.getcwd()}\n"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [
                TextContent(
                    type="text", text=f"Test error: {str(e)}\n{traceback.format_exc()}"
                )
            ]

    async def execute_mathematica(
        self, code: str, format_latex: bool = True
    ) -> List[TextContent]:
        """Execute Mathematica code with full error handling and formatting."""
        try:
            # Check if wolframscript is available
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

            # FIX: Wrap code to ensure output is captured
            wrapped_code = f"""
(* MCP Server Execution *)
{context}
result = {code};
Print[OutputForm[result]];
result
"""

            logger.info(f"Executing wrapped Mathematica code: {wrapped_code[:200]}...")

            # FIX: Use explicit options for better output control
            cmd_args = [
                "wolframscript",
                "-code",
                wrapped_code,
                "-print",
                "all",  # Print all outputs
                "-format",
                "OutputForm",  # Force text output
                "-charset",
                "UTF8",  # Ensure proper encoding
            ]

            # FIX: Ensure proper environment
            env = os.environ.copy()
            # Ensure Mathematica path is at the front
            mathematica_path = "/Applications/Mathematica.app/Contents/MacOS"
            current_path = env.get("PATH", "")
            if mathematica_path not in current_path:
                env["PATH"] = f"{mathematica_path}:{current_path}"

            # Add critical environment variables if not present
            if "HOME" not in env:
                env["HOME"] = os.path.expanduser("~")
            if "USER" not in env:
                import pwd

                env["USER"] = pwd.getpwuid(os.getuid()).pw_name

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.session.temp_dir,  # Use session temp dir as working directory
            )

            logger.info(f"Process created with PID: {process.pid}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
                logger.info(f"Process completed with return code: {process.returncode}")
                logger.info(f"Raw stdout length: {len(stdout)} bytes")
                logger.info(f"Raw stderr length: {len(stderr)} bytes")

                # Log first 500 chars of output for debugging
                if stdout:
                    logger.info(
                        f"Stdout preview: {stdout.decode('utf-8', errors='ignore')[:500]}"
                    )
                if stderr:
                    logger.info(
                        f"Stderr preview: {stderr.decode('utf-8', errors='ignore')[:500]}"
                    )

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
                result = stdout.decode("utf-8", errors="ignore").strip()

                # FIX: If still empty, try alternative approach
                if not result:
                    logger.warning(
                        "Empty result from wolframscript, trying alternative approach"
                    )

                    # Try using a temporary file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".m", delete=False
                    ) as f:
                        f.write(f"{context}\n{code}\n")
                        temp_file = f.name

                    try:
                        alt_process = await asyncio.create_subprocess_exec(
                            "wolframscript",
                            "-file",
                            temp_file,
                            "-print",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                            cwd=self.session.temp_dir,
                        )
                        alt_stdout, alt_stderr = await alt_process.communicate()
                        result = alt_stdout.decode("utf-8", errors="ignore").strip()

                        if not result:
                            logger.error(
                                f"Still empty. Alt stderr: {alt_stderr.decode('utf-8', errors='ignore')}"
                            )
                    finally:
                        os.unlink(temp_file)

                if not result:
                    # Provide diagnostic information
                    stderr_content = stderr.decode("utf-8", errors="ignore")
                    return [
                        TextContent(
                            type="text",
                            text=f"**Error:** Empty output from wolframscript\n"
                            f"Stderr: {stderr_content}\n"
                            f"Working directory: {self.session.temp_dir}\n"
                            f"Try using the test_wolframscript tool to diagnose the issue.",
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
                error = stderr.decode("utf-8", errors="ignore").strip()
                error = MathematicaValidator.sanitize_output(error)

                return [TextContent(type="text", text=f"**Execution Error:**\n{error}")]

        except Exception as e:
            logger.error(f"Unexpected error in execute_mathematica: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"**System Error:** {str(e)}")]

    async def get_session_info(self) -> List[TextContent]:
        """Get information about the current session."""
        info = [
            f"**Session ID:** {self.session.session_id}",
            f"**Variables:** {len(self.session.variables)}",
            f"**History entries:** {len(self.session.history)}",
            f"**Temp directory:** {self.session.temp_dir}",
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
