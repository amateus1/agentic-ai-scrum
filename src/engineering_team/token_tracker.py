"""
token_tracker.py
🚧 Public Demo Version (Stubbed)

⚠️ Important: Detailed token usage & cost tracking logic has been 
**stubbed** in this public demo.  

The public repo demonstrates the **CrewAI agent setup**, 
but production-ready token attribution (multi-provider parsing, 
per-agent usage, cost reporting) is only available via **OptimOps.ai**.  

📩 Contact: al@optimops.ai
"""

import logging
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

class EnhancedTokenUsageCallbackHandler(BaseCallbackHandler):
    """Stubbed callback handler for demo purposes only."""

    def __init__(self):
        self.usage_data = {}
        self.current_agent = "Unknown"

    def set_current_agent(self, agent_name: str):
        self.current_agent = agent_name

    def on_llm_end(self, response, **kwargs):
        """
        🚧 Stubbed: In production this would parse token usage from multiple LLM providers.
        """
        logger.info(f"[STUB] Token usage tracking disabled for {self.current_agent}")

    def get_usage_summary(self):
        """
        Return stubbed usage summary.
        """
        return {"status": "stubbed", "message": "Token tracking not available in demo mode"}
