"""
Mathematica Model Context Protocol (MCP) Server - No TeXForm Version
Uses OutputForm only, letting Claude handle LaTeX formatting
"""

import asyncio
import os
import re
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
            # Clean the variable value to avoid syntax issues
            clean_value = var_value.replace('\\"', '"').replace("\\n", "\n")
            # Safely reconstruct variable assignments
            context_code.append(f"{var_name} = {clean_value};")

        return "\n".join(context_code)

    def clear_bad_variables(self):
        """Remove variables that might contain syntax errors."""
        bad_vars = []
        for var_name, var_value in self.variables.items():
            if '\\"' in var_value or "ToExpression::sntx" in var_value:
                bad_vars.append(var_name)

        for var_name in bad_vars:
            del self.variables[var_name]
            logger.info(f"Removed problematic variable: {var_name}")

        if bad_vars:
            logger.info(f"Cleared {len(bad_vars)} problematic variables from session")

    def update_variables(self, code: str, result: str):
        """Extract and update session variables from executed code."""
        # Only update variables if the execution was successful (no error messages)
        if "Error" in result or "ToExpression::sntx" in result or "Syntax::" in result:
            logger.info("Skipping variable update due to execution error")
            return

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
    """Handles formatting of Mathematica output using OutputForm only."""

    @staticmethod
    def format_result(result: str) -> str:
        return f"**Result:**\n{result}"


class MathematicaMCPServer:
    """Main MCP server for Mathematica integration."""

    def __init__(self):
        self.server = Server("mathematica-server")
        # Add server info to help Claude understand capabilities
        self.server.server_info = {
            "name": "Mathematica MCP Server",
            "version": "1.1.0",
            "description": "Advanced mathematical computation server powered by Wolfram Mathematica. Provides symbolic mathematics, calculus, algebra, statistics, plotting, and numerical analysis capabilities. Use for any mathematical task that goes beyond basic arithmetic. Outputs in OutputForm for Claude to format.",
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
                    description=(
                        "Execute Mathematica/Wolfram Language code for mathematical computations, symbolic math, calculus, algebra, "
                        "differential equations, statistics, plotting, and numerical analysis. "
                        "Use this tool for any advanced mathematical task.\n\n"
                        "**When generating Mathematica code:**\n"
                        '- Use plain double quotes (") for string literals, not escaped quotes (\\").\n'
                        "- Do not escape quotes inside Mathematica code unless absolutely necessary.\n"
                        "- End each Mathematica statement with a semicolon (;).\n"
                        "- Ensure the code is valid Mathematica syntax and can be run directly in a Mathematica notebook or via wolframscript.\n"
                        '- If you need to print text, use Print["text"], not Print[\\"text\\"].'
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Mathematica/Wolfram Language code to execute. Examples: 'Solve[x^2 + 2*x + 1 == 0, x]', 'Integrate[x^2, x]', 'Plot[Sin[x], {x, 0, 2*Pi}]', 'Factor[x^4 - 1]'",
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
                return await self.execute_mathematica(arguments["code"])
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
            result += "\nEnvironment:\n"
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

    def preprocess_mathematica_code(self, code: str) -> str:
        """Preprocess Mathematica code to fix common LLM escaping issues."""
        # Replace \" with "
        code = code.replace('\\"', '"')
        # Optionally, replace double-escaped newlines with real newlines
        code = code.replace("\\n", "\n")
        return code

    def is_multi_statement_code(self, code: str) -> bool:
        """
        Determine if code contains multiple statements that should be executed directly
        rather than wrapped in a single result assignment.
        """
        code = code.strip()

        # Check for multiple statements separated by semicolons
        if ";" in code:
            # Split by semicolon and check if we have multiple non-empty statements
            statements = [stmt.strip() for stmt in code.split(";") if stmt.strip()]
            if len(statements) > 1:
                return True

        # Check for Print statements (should be executed directly)
        if code.startswith("Print[") or "Print[" in code:
            return True

        # Check for plotting functions (should be executed directly)
        plot_functions = [
            "Plot[",
            "Plot3D[",
            "ListPlot[",
            "ContourPlot[",
            "ParametricPlot[",
        ]
        if any(func in code for func in plot_functions):
            return True

        # Check for DSolve, NDSolve and other functions that might produce complex output
        complex_functions = ["DSolve[", "NDSolve[", "Solve["]
        if any(func in code for func in complex_functions):
            # For these, check if they're already wrapped in Print or assigned
            if not (code.startswith("Print[") or "=" in code.split(";")[0]):
                return False  # Let these be wrapped for output

        return False

    async def execute_mathematica(self, code: str) -> List[TextContent]:
        """Execute Mathematica code with OutputForm formatting only."""
        try:
            # Preprocess code to fix LLM escaping issues
            code = self.preprocess_mathematica_code(code)

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

            # Clean any problematic variables from previous sessions
            self.session.clear_bad_variables()

            # Prepare code with session context
            context = self.session.get_context_variables()

            # Determine if code is multi-statement or single expression
            is_multi_statement = self.is_multi_statement_code(code)

            if is_multi_statement:
                # Multi-statement code - execute directly
                wrapped_code = f"""
(* MCP Server Execution *)
{context}
{code}
"""
            else:
                # Single expression - wrap in result assignment for output
                wrapped_code = f"""
(* MCP Server Execution *)
{context}
result = {code};
Print[OutputForm[result]];
"""

            logger.info(f"Executing wrapped Mathematica code: {wrapped_code[:200]}...")

            # Use OutputForm explicitly for all outputs
            cmd_args = [
                "wolframscript",
                "-code",
                wrapped_code,
                "-print",
                "all",
                "-format",
                "OutputForm",
                "-charset",
                "UTF8",
            ]

            env = os.environ.copy()
            mathematica_path = "/Applications/Mathematica.app/Contents/MacOS"
            current_path = env.get("PATH", "")
            if mathematica_path not in current_path:
                env["PATH"] = f"{mathematica_path}:{current_path}"
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
                cwd=self.session.temp_dir,
            )

            logger.info(f"Process created with PID: {process.pid}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
                logger.info(f"Process completed with return code: {process.returncode}")
                logger.info(f"Raw stdout length: {len(stdout)} bytes")
                logger.info(f"Raw stderr length: {len(stderr)} bytes")
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
                if not result:
                    logger.warning(
                        "Empty result from wolframscript, trying alternative approach"
                    )
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
                result = MathematicaValidator.sanitize_output(result)
                logger.info(f"Sanitized Mathematica output: '{result}'")
                self.session.update_variables(code, result)
                self.session.add_to_history(code, result)
                formatted_output = MathematicaFormatter.format_result(result)
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
