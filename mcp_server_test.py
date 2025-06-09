#!/usr/bin/env python3
"""
Test script for the Mathematica MCP Server
Tests all major functionality including validation, execution, and session management.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any


# Color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class TestResult:
    def __init__(
        self, name: str, passed: bool, message: str = "", execution_time: float = 0.0
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.execution_time = execution_time


class MathematicaServerTester:
    def __init__(self, server_script_path: str):
        self.server_script_path = Path(server_script_path)
        self.results: List[TestResult] = []
        self.server_process = None

    def print_status(self, message: str, status: str = "INFO"):
        colors = {
            "INFO": Colors.BLUE,
            "PASS": Colors.GREEN,
            "FAIL": Colors.RED,
            "WARN": Colors.YELLOW,
        }
        color = colors.get(status, Colors.BLUE)
        print(f"{color}[{status}]{Colors.ENDC} {message}")

    async def test_prerequisites(self) -> TestResult:
        """Test if all prerequisites are available."""
        start_time = time.time()

        try:
            # Check if server script exists
            if not self.server_script_path.exists():
                return TestResult(
                    "Prerequisites",
                    False,
                    f"Server script not found: {self.server_script_path}",
                    time.time() - start_time,
                )

            # Check if wolframscript is available
            try:
                process = await asyncio.create_subprocess_exec(
                    "wolframscript",
                    "-version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    return TestResult(
                        "Prerequisites",
                        False,
                        "wolframscript not working properly",
                        time.time() - start_time,
                    )

                version_info = stdout.decode("utf-8").strip()

            except FileNotFoundError:
                return TestResult(
                    "Prerequisites",
                    False,
                    "wolframscript not found in PATH",
                    time.time() - start_time,
                )

            # Check Python imports
            try:
                import mcp
                from mcp.server import Server
                from mcp.types import Tool, TextContent
            except ImportError as e:
                return TestResult(
                    "Prerequisites",
                    False,
                    f"Missing Python dependencies: {e}",
                    time.time() - start_time,
                )

            return TestResult(
                "Prerequisites",
                True,
                f"All prerequisites available. Wolfram version: {version_info[:50]}...",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Prerequisites",
                False,
                f"Unexpected error: {e}",
                time.time() - start_time,
            )

    async def test_server_import(self) -> TestResult:
        """Test if the server script can be imported without errors."""
        start_time = time.time()

        try:
            # Add the server directory to Python path temporarily
            server_dir = self.server_script_path.parent
            sys.path.insert(0, str(server_dir))

            # Try to import the main classes
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "mathematica_server", self.server_script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if main classes exist
            required_classes = [
                "MathematicaMCPServer",
                "MathematicaSession",
                "MathematicaValidator",
                "MathematicaFormatter",
            ]
            for class_name in required_classes:
                if not hasattr(module, class_name):
                    return TestResult(
                        "Server Import",
                        False,
                        f"Missing required class: {class_name}",
                        time.time() - start_time,
                    )

            return TestResult(
                "Server Import",
                True,
                "Server script imports successfully",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Server Import", False, f"Import error: {e}", time.time() - start_time
            )
        finally:
            # Remove from path
            if str(server_dir) in sys.path:
                sys.path.remove(str(server_dir))

    async def test_validation_system(self) -> TestResult:
        """Test the code validation system."""
        start_time = time.time()

        try:
            # Import validation class
            server_dir = self.server_script_path.parent
            sys.path.insert(0, str(server_dir))

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "mathematica_server", self.server_script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            validator = module.MathematicaValidator()

            # Test valid code
            valid_code = "x = 5; y = x^2; Solve[y == 25, x]"
            is_valid, error = validator.validate_code(valid_code)
            if not is_valid:
                return TestResult(
                    "Validation System",
                    False,
                    f"Valid code rejected: {error}",
                    time.time() - start_time,
                )

            # Test dangerous code
            dangerous_codes = [
                'DeleteFile["test.txt"]',
                'Run["rm -rf /"]',
                'Import["http://malicious.com"]',
            ]

            for dangerous_code in dangerous_codes:
                is_valid, error = validator.validate_code(dangerous_code)
                if is_valid:
                    return TestResult(
                        "Validation System",
                        False,
                        f"Dangerous code not caught: {dangerous_code}",
                        time.time() - start_time,
                    )

            return TestResult(
                "Validation System",
                True,
                "Validation system working correctly",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Validation System",
                False,
                f"Validation test error: {e}",
                time.time() - start_time,
            )
        finally:
            if str(server_dir) in sys.path:
                sys.path.remove(str(server_dir))

    async def test_basic_mathematica_execution(self) -> TestResult:
        """Test basic Mathematica code execution."""
        start_time = time.time()

        try:
            # Simple test cases
            test_cases = [
                ("2 + 2", "4"),
                ("Sqrt[16]", "4"),
                ("Factor[x^2 - 1]", ["(-1 + x)*(1 + x)", "(-1 + x) (1 + x)"]),
            ]

            for code, expected_pattern in test_cases:
                process = await asyncio.create_subprocess_exec(
                    "wolframscript",
                    "-code",
                    code,
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
                    return TestResult(
                        "Basic Execution",
                        False,
                        f"Timeout executing: {code}",
                        time.time() - start_time,
                    )

                if process.returncode != 0:
                    return TestResult(
                        "Basic Execution",
                        False,
                        f"Failed to execute: {code}. Error: {stderr.decode('utf-8')}",
                        time.time() - start_time,
                    )

                result = stdout.decode("utf-8").strip()
                if isinstance(expected_pattern, list):
                    if not any(p in result for p in expected_pattern):
                        return TestResult(
                            "Basic Execution",
                            False,
                            f"Unexpected result for {code}: got '{result}', expected one of {expected_pattern}",
                            time.time() - start_time,
                        )
                else:
                    if expected_pattern not in result:
                        return TestResult(
                            "Basic Execution",
                            False,
                            f"Unexpected result for {code}: got '{result}', expected pattern '{expected_pattern}'",
                            time.time() - start_time,
                        )

            return TestResult(
                "Basic Execution",
                True,
                "Basic Mathematica execution working",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Basic Execution",
                False,
                f"Execution test error: {e}",
                time.time() - start_time,
            )

    async def test_latex_formatting(self) -> TestResult:
        """Test LaTeX formatting functionality."""
        start_time = time.time()

        try:
            # Test LaTeX conversion
            test_expression = "x^2 + 2*x + 1"
            latex_code = f"TeXForm[{test_expression}]"

            process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "-code",
                latex_code,
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
                return TestResult(
                    "LaTeX Formatting",
                    False,
                    f"Timeout during LaTeX conversion",
                    time.time() - start_time,
                )

            if process.returncode != 0:
                return TestResult(
                    "LaTeX Formatting",
                    False,
                    f"LaTeX conversion failed: {stderr.decode('utf-8')}",
                    time.time() - start_time,
                )

            latex_result = stdout.decode("utf-8").strip()

            # Check if we got a reasonable LaTeX output
            if not latex_result or len(latex_result) < 3:
                return TestResult(
                    "LaTeX Formatting",
                    False,
                    f"LaTeX output too short: '{latex_result}'",
                    time.time() - start_time,
                )

            return TestResult(
                "LaTeX Formatting",
                True,
                f"LaTeX formatting working. Sample: {latex_result[:50]}...",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "LaTeX Formatting",
                False,
                f"LaTeX test error: {e}",
                time.time() - start_time,
            )

    async def test_session_management(self) -> TestResult:
        """Test session management functionality."""
        start_time = time.time()

        try:
            # Import session class
            server_dir = self.server_script_path.parent
            sys.path.insert(0, str(server_dir))

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "mathematica_server", self.server_script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Create a session
            session = module.MathematicaSession()

            # Test variable tracking
            session.update_variables("x = 5", "5")
            session.update_variables("y = x^2", "25")

            if len(session.variables) != 2:
                return TestResult(
                    "Session Management",
                    False,
                    f"Variable tracking failed. Expected 2 variables, got {len(session.variables)}",
                    time.time() - start_time,
                )

            # Test history
            session.add_to_history("test1", "result1")
            session.add_to_history("test2", "result2")

            if len(session.history) != 2:
                return TestResult(
                    "Session Management",
                    False,
                    f"History tracking failed. Expected 2 entries, got {len(session.history)}",
                    time.time() - start_time,
                )

            # Test context generation
            context = session.get_context_variables()
            if "x = 5" not in context or "y = x^2" not in context:
                return TestResult(
                    "Session Management",
                    False,
                    f"Context generation failed: {context}",
                    time.time() - start_time,
                )

            # Cleanup
            session.cleanup()

            return TestResult(
                "Session Management",
                True,
                "Session management working correctly",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Session Management",
                False,
                f"Session test error: {e}",
                time.time() - start_time,
            )
        finally:
            if str(server_dir) in sys.path:
                sys.path.remove(str(server_dir))

    async def test_server_startup(self) -> TestResult:
        """Test if the server can start up without errors."""
        start_time = time.time()

        try:
            # Try to start the server process
            self.server_process = await asyncio.create_subprocess_exec(
                "python",
                str(self.server_script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Give it a moment to start
            await asyncio.sleep(2)

            # Check if it's still running
            if self.server_process.returncode is not None:
                # Process has already terminated
                stdout, stderr = await self.server_process.communicate()
                return TestResult(
                    "Server Startup",
                    False,
                    f"Server terminated immediately. Stderr: {stderr.decode('utf-8')[:200]}",
                    time.time() - start_time,
                )

            # Try to terminate gracefully
            self.server_process.terminate()
            await asyncio.sleep(1)

            if self.server_process.returncode is None:
                # Force kill if still running
                self.server_process.kill()
                await self.server_process.wait()

            return TestResult(
                "Server Startup",
                True,
                "Server starts and runs successfully",
                time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                "Server Startup",
                False,
                f"Server startup error: {e}",
                time.time() - start_time,
            )

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        self.print_status("Starting Mathematica MCP Server Tests", "INFO")
        self.print_status("=" * 50, "INFO")

        # Define test sequence
        tests = [
            ("Prerequisites Check", self.test_prerequisites),
            ("Server Import", self.test_server_import),
            ("Validation System", self.test_validation_system),
            ("Basic Execution", self.test_basic_mathematica_execution),
            ("LaTeX Formatting", self.test_latex_formatting),
            ("Session Management", self.test_session_management),
            ("Server Startup", self.test_server_startup),
        ]

        total_tests = len(tests)
        passed_tests = 0

        for test_name, test_func in tests:
            self.print_status(f"Running: {test_name}", "INFO")

            try:
                result = await test_func()
                self.results.append(result)

                if result.passed:
                    self.print_status(
                        f"✓ {test_name}: {result.message} ({result.execution_time:.2f}s)",
                        "PASS",
                    )
                    passed_tests += 1
                else:
                    self.print_status(
                        f"✗ {test_name}: {result.message} ({result.execution_time:.2f}s)",
                        "FAIL",
                    )

            except Exception as e:
                error_result = TestResult(
                    test_name, False, f"Test framework error: {e}"
                )
                self.results.append(error_result)
                self.print_status(f"✗ {test_name}: {error_result.message}", "FAIL")

        # Summary
        self.print_status("=" * 50, "INFO")
        self.print_status(
            f"Test Results: {passed_tests}/{total_tests} passed",
            "PASS" if passed_tests == total_tests else "WARN",
        )

        # Detailed results
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            self.print_status("\nFailed Tests:", "FAIL")
            for result in failed_tests:
                self.print_status(f"  • {result.name}: {result.message}", "FAIL")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": len(failed_tests),
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.results,
        }


async def main():
    """Main test runner."""
    if len(sys.argv) != 2:
        print(
            f"{Colors.RED}Usage: python test_mathematica_server.py <path_to_server_script>{Colors.ENDC}"
        )
        print(
            f"{Colors.YELLOW}Example: python test_mathematica_server.py ./mathematica_server.py{Colors.ENDC}"
        )
        sys.exit(1)

    server_script_path = sys.argv[1]

    tester = MathematicaServerTester(server_script_path)
    results = await tester.run_all_tests()

    # Exit with appropriate code
    if results["success_rate"] == 1.0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! 🎉{Colors.ENDC}")
        sys.exit(0)
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Please check the issues above.{Colors.ENDC}"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
