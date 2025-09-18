# src/engineering_team/token_tracker.py

import json
import logging
from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
import re

logger = logging.getLogger(__name__)

class EnhancedTokenUsageCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler that works with multiple LLM providers"""
    
    def __init__(self):
        self.usage_data = {}
        self.current_agent = "Unknown"
    
    def set_current_agent(self, agent_name: str):
        """Set the current agent for token attribution"""
        self.current_agent = agent_name
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture token usage when LLM calls complete"""
        try:
            # Try different methods to extract token usage based on provider
            token_usage = self._extract_token_usage_provider_specific(response)
            
            if token_usage:
                model = token_usage.get('model', 'unknown')
                self._store_usage_data(model, token_usage)
                
        except Exception as e:
            logger.debug(f"Failed to capture token usage: {e}")
    
    def _extract_token_usage_provider_specific(self, response: Any) -> Dict[str, Any]:
        """Extract token usage based on LLM provider"""
        try:
            # Method 1: Check if response has usage data directly
            if hasattr(response, 'usage'):
                usage = response.usage
                model = getattr(response, 'model', 'unknown')
                return {
                    'input_tokens': getattr(usage, 'prompt_tokens', 0),
                    'output_tokens': getattr(usage, 'completion_tokens', 0),
                    'model': model
                }
            
            # Method 2: Check llm_output (OpenAI format)
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage', {})
                model = response.llm_output.get('model', 'unknown')
                return {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'model': model
                }
            
            # Method 3: Check response_metadata (Anthropic format)
            if hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get('usage', {})
                model = response.response_metadata.get('model', 'unknown')
                return {
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'model': model
                }
            
            # Method 4: Try to parse from raw content
            raw_content = getattr(response, 'content', '') or str(response)
            return self._parse_usage_from_content(raw_content)
            
        except Exception as e:
            logger.debug(f"Could not extract token usage: {e}")
            return {}
    
    def _parse_usage_from_content(self, content: str) -> Dict[str, Any]:
        """Parse token usage from raw content string"""
        try:
            # Look for common patterns in LLM responses
            patterns = [
                # OpenAI pattern
                (r'"prompt_tokens":\s*(\d+)', r'"completion_tokens":\s*(\d+)', r'"model":\s*"([^"]+)"'),
                # Anthropic pattern
                (r'"input_tokens":\s*(\d+)', r'"output_tokens":\s*(\d+)', r'"model":\s*"([^"]+)"'),
                # Generic pattern
                (r'prompt_tokens[=:]\s*(\d+)', r'completion_tokens[=:]\s*(\d+)', r'model[=:]\s*"([^"]+)"')
            ]
            
            for input_pattern, output_pattern, model_pattern in patterns:
                input_match = re.search(input_pattern, content)
                output_match = re.search(output_pattern, content)
                model_match = re.search(model_pattern, content)
                
                if input_match and output_match:
                    return {
                        'input_tokens': int(input_match.group(1)),
                        'output_tokens': int(output_match.group(1)),
                        'model': model_match.group(1) if model_match else 'unknown'
                    }
                    
        except Exception:
            pass
            
        return {}
    
    def _store_usage_data(self, model: str, token_usage: Dict[str, Any]):
        """Store token usage data with agent attribution"""
        if self.current_agent not in self.usage_data:
            self.usage_data[self.current_agent] = {}
        
        # Clean model name (remove provider prefix)
        clean_model = self._clean_model_name(model)
        
        if clean_model not in self.usage_data[self.current_agent]:
            self.usage_data[self.current_agent][clean_model] = {
                "input_tokens": 0,
                "output_tokens": 0
            }
        
        # Update token counts
        self.usage_data[self.current_agent][clean_model]["input_tokens"] += token_usage.get('input_tokens', 0)
        self.usage_data[self.current_agent][clean_model]["output_tokens"] += token_usage.get('output_tokens', 0)
        
        logger.info(f"Token usage for {self.current_agent} - {clean_model}: {token_usage}")
    
    def _clean_model_name(self, model: str) -> str:
        """Remove provider prefixes from model names"""
        # Remove common provider prefixes
        prefixes = ['openai/', 'anthropic/', 'deepseek/', 'qwen/', 'azure/', 'aws/']
        clean_model = model
        for prefix in prefixes:
            if clean_model.startswith(prefix):
                clean_model = clean_model[len(prefix):]
        return clean_model
