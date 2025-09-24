"""
main.py
🚧 Public Demo Version (Stubbed)

⚠️ Important: Some cost tracking and production integration logic has been 
**stubbed** in this public demo.

You can still run local demos (`python main.py`) to see CrewAI agents in action.  
The full webhook → ADO → CrewAI → W&B integration is only available 
via **OptimOps.ai** for client engagements.

📩 Contact: al@optimops.ai
"""

import logging
from crew import start_crew

def _distribute_global_token_usage_by_agent():
    """
    🚧 Stubbed: In production, this would distribute token usage and costs by agent role.
    """
    logging.info("[STUB] Token usage distribution disabled in demo mode.")
    return {}

def run_demo():
    """
    Run a local demo of the CrewAI team.
    """
    logging.info("[DEMO] Starting CrewAI demo run (stubbed integration).")
    crew_result = start_crew()  # Still runs basic crew logic
    logging.info(f"[DEMO] Crew finished with result: {crew_result}")
    return crew_result

if __name__ == "__main__":
    run_demo()
