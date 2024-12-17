from typing import Dict, List, Optional
import anthropic
from mcpcli.environment import load_environment

class AnthropicProvider:
    """Anthropic provider implementation for the MCP CLI."""
    
    def __init__(self):
        self.env = load_environment()
        self.client = anthropic.Anthropic(api_key=self.env.get("ANTHROPIC_API_KEY"))
        self.default_model = "claude-3-sonnet-20240229"

    async def generate_response(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict:
        """Generate a response using the Anthropic API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                role = "assistant" if msg["role"] == "assistant" else "user"
                anthropic_messages.append({
                    "role": role,
                    "content": msg["content"]
                })

            # Create the chat completion
            response = self.client.messages.create(
                model=model or self.default_model,
                messages=anthropic_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return {
                "content": response.content[0].text,
                "finish_reason": response.stop_reason,
                "model": response.model,
                "provider": "anthropic"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "provider": "anthropic",
                "isError": True
            }

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for Anthropic's API."""
        formatted = []
        for msg in messages:
            if msg["role"] == "system":
                # Handle system messages as user messages with special prefix
                formatted.append({
                    "role": "user",
                    "content": f"System: {msg['content']}"
                })
            else:
                formatted.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        return formatted