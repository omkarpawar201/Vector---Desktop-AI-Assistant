"""
Google Gemini API Client Integration for Vector Desktop AI Assistant.
Handles multi-step autonomous tool reasoning loops with strict security guardrails.
"""

from typing import Any, Dict, List, Optional
from app.config.constants import MAX_TOOL_CALLS
from app.config.settings import Settings, get_settings
from app.core.privacy import PrivacyFilter
from app.tools.executor import SafeToolExecutor
from app.tools.registry import ToolRegistry, get_tool_registry


class GeminiClient:
    """
    Cloud Gemini API integration supporting autonomous multi-tool execution loops.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[ToolRegistry] = None,
        executor: Optional[SafeToolExecutor] = None
    ):
        self.settings = settings or get_settings()
        self.registry = registry or get_tool_registry()
        self.executor = executor or SafeToolExecutor(registry=self.registry)

    def _convert_tools_to_gemini_declarations(self) -> List[Dict[str, Any]]:
        """
        Converts ToolRegistry OpenAPI schemas into Gemini API FunctionDeclarations format.
        """
        schemas = self.registry.export_schemas()
        declarations = []
        for s in schemas:
            declarations.append({
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"]
            })
        return declarations

    def execute_reasoning_loop(
        self,
        user_query: str,
        confirmed_by_user: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a multi-tool reasoning loop with Gemini API.
        Enforces MAX_TOOL_CALLS = 10 limit to prevent infinite loops or lockups.
        """
        if not self.settings.is_gemini_available:
            return {
                "success": False,
                "response": "Gemini unavailable. Cloud fallback is disabled or API key is missing.",
                "tool_calls": []
            }

        # 1. Sanitize user query through PrivacyFilter
        sanitized_query = PrivacyFilter.sanitize_text(user_query)

        tool_calls_executed: List[Dict[str, Any]] = []
        call_count = 0

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.settings.gemini_api_key)
            tools_spec = self._convert_tools_to_gemini_declarations()

            system_instruction = (
                "You are Vector, a local desktop AI assistant. "
                "You control the user's computer via provided function tools. "
                "Use the available tools to satisfy the user's request. "
                "Never output unverified claims. Return clear, concise responses."
            )

            chat = client.chats.create(
                model=self.settings.gemini_model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )

            response = chat.send_message(sanitized_query)

            # Autonomous Multi-Tool Loop
            while response.function_calls and call_count < MAX_TOOL_CALLS:
                call_count += 1
                function_call = response.function_calls[0]
                tool_name = function_call.name
                raw_args = dict(function_call.args) if function_call.args else {}

                # Execute tool via SafeToolExecutor
                res, sm = self.executor.execute_tool(
                    tool_name=tool_name,
                    raw_args=raw_args,
                    confirmed_by_user=confirmed_by_user
                )

                tool_calls_executed.append({
                    "step": call_count,
                    "tool": tool_name,
                    "arguments": raw_args,
                    "success": res.success,
                    "message": res.message,
                    "data": res.data
                })

                if not res.success and res.error == "CONFIRMATION_REQUIRED":
                    # Require user confirmation modal for dangerous/confirm tools
                    return {
                        "success": False,
                        "requires_confirmation": True,
                        "tool_name": tool_name,
                        "arguments": raw_args,
                        "response": res.message,
                        "tool_calls": tool_calls_executed
                    }

                # Format tool output response back to Gemini
                tool_response_content = {
                    "output": res.data or {},
                    "message": res.message,
                    "success": res.success,
                    "error": res.error
                }

                # Send tool execution result back to Gemini for next reasoning step
                response = chat.send_message(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_response_content}
                    )
                )

            final_text = response.text or "Completed request successfully."
            return {
                "success": True,
                "response": final_text,
                "tool_calls": tool_calls_executed,
                "total_tool_calls": call_count
            }

        except ImportError:
            # Fallback if google-genai is not installed or loading lazily
            return {
                "success": False,
                "response": "Google GenAI SDK not installed. Please run pip install google-genai.",
                "tool_calls": []
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Gemini API execution error: {str(e)}",
                "tool_calls": tool_calls_executed
            }
