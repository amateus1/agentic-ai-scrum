# src/engineering_team/ado_client.py

import os
import logging
import requests
from datetime import datetime
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Environment variables
ADO_ORG = os.getenv("ADO_ORG")             # e.g., 'yourorgname'
ADO_PROJECT = os.getenv("ADO_PROJECT")     # e.g., 'yourproject'
ADO_PAT = os.getenv("ADO_PAT")             # stored in .env
ADO_USER = os.getenv("ADO_USER")           # e.g., crewai-lead@optimops.ai
WEAVE_URL_BASE = os.getenv("WEAVE_URL_BASE", "https://wandb.ai")
APP_URL_BASE = os.getenv("APP_URL_BASE", "https://your-app-domain.com")

# Base URL construction
BASE_URL = f"https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT}/_apis"

# Authentication
auth = HTTPBasicAuth("", ADO_PAT)

class ADOClientError(Exception):
    """Custom exception for Azure DevOps client errors"""
    pass

def _make_request(method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Helper function to make HTTP requests with error handling
    """
    try:
        logger.debug(f"Making {method} request to: {url}")
        response = requests.request(method, url, auth=auth, **kwargs)
        response.raise_for_status()
        
        # Try to parse JSON, but return empty dict if it fails
        try:
            return response.json()
        except:
            return {}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Azure DevOps API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text[:500]}")
        raise ADOClientError(f"Azure DevOps API request failed: {e}") from e

def post_comment(work_item_id: int, message: str) -> Dict[str, Any]:
    """
    Post a comment to an Azure DevOps work item
    """
    url = f"{BASE_URL}/wit/workItems/{work_item_id}/comments?api-version=7.0-preview.3"
    payload = {"text": message}
    headers = {"Content-Type": "application/json"}
    
    logger.info(f"Posting comment to work item {work_item_id}")
    
    return _make_request("post", url, headers=headers, json=payload)

def post_execution_started(work_item_id: int, task_description: str) -> Dict[str, Any]:
    """
    Post comment when execution starts
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"""🔧 **Execution Started** / 执行已开始

    
📅 {timestamp}
📋 Task: {task_description}
    
⏳ Agents are now working on the task... / 智能体正在处理任务... 工程 Crew AI 已初始化
    
_Engineering Crew AI initialized_"""
    
    return post_comment(work_item_id, message)
def _fmt_usd(amount: float) -> str:
    """Format USD currency"""
    return f"${amount:.4f}"

def post_crew_completed(
    work_item_id: int, 
    total_time: float, 
    llms_used: list = None,
    by_agent: dict = None, 
    by_llm: dict = None, 
    total_cost: float = None,
    weave_url: str = "", 
    app_url: str = ""
) -> Dict[str, Any]:
    """
    Post completion comment with optional cost breakdown
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    if llms_used and by_agent and by_llm and total_cost is not None:
        # Detailed version with cost breakdown
        llms_list = "\n".join([f"    - {m}" for m in sorted(llms_used)])
        
        by_agent_str = ""
        for agent, data in by_agent.items():
            if isinstance(data, dict) and 'models' in data:
                # Get the main model and cost (assuming one model per agent)
                models = data['models']
                if models:
                    model_name, model_cost = next(iter(models.items()))
                    by_agent_str += f"    - {agent}: {model_name}: {_fmt_usd(model_cost)}\n"
            else:
                # Fallback for simple cost format
                by_agent_str += f"    - {agent}: {_fmt_usd(data)}\n"
        
        by_llm_str = "\n".join([f"    - {model}: {_fmt_usd(cost)}" for model, cost in by_llm.items()])
        
        # Handles both run URLs and direct weave URLs
        if weave_url:
            if "/runs/" in weave_url:
                run_id = weave_url.split("/runs/")[-1]
                weave_url = f"https://wandb.ai/amateus1-optimops-ai/crewai-ado-integration/weave/traces?view=traces_{run_id}"
            elif "wandb.ai" in weave_url and "/weave/" not in weave_url:
                # Convert basic wandb URL to weave traces URL
                weave_url = f"https://wandb.ai/amateus1-optimops-ai/crewai-ado-integration/weave/traces"
                    
        message = f"""✅ **Work Completed Successfully / 工作已成功完成**

📅 {timestamp}
⏱️ Total time / 总耗时: {total_time:.2f}s

🧠 **LLMs Used / 使用的 LLM:**
{llms_list}

💰 **Cost Breakdown / 成本明细:**

👥 **By Agent / 按角色:**
{by_agent_str}
🧠 **By LLM (Total) / 按 LLM（总计）:**
{by_llm_str}

💵 **Total Cost / 总成本: {_fmt_usd(total_cost)}**

🔗 View Weave Trace:     {weave_url}
🌐 View Application / 查看应用:       {app_url}

🎯 All tasks completed successfully / 所有任务均已成功完成

_Engineering Crew AI execution finished / 工程 Crew AI 执行结束_"""
        message = message.replace('\n', '  \n')  # ✅ critical for line breaks in ADO
    else:
        # Simple version (backward compatible)
        url_section = ""
        if weave_url:
            url_section += f"🔗 View Weave Trace:     {weave_url}\n"
        if app_url:
            url_section += f"🌐 View Application / 查看应用:       {app_url}\n"
        if url_section:
            url_section = "\n" + url_section
        
        message = f"""✅ **Work Completed Successfully**
        
📅 {timestamp}
⏱️ Total time: {total_time:.2f}s
{url_section}
🎯 All tasks completed successfully

_Engineering Crew AI execution finished_"""
        message = message.replace('\n', '  \n')  # ✅ fix for simple version too

    return post_comment(work_item_id, message.replace("\n", "<br>"))

def post_crew_initializing(work_item_id: int) -> Dict[str, Any]:
    """Post comment when crew is initializing"""
    message = """🔄 Starting CrewAI execution... / 正在启动 CrewAI 执行..."""
    return post_comment(work_item_id, message)

def post_team_initializing(work_item_id: int) -> Dict[str, Any]:
    """Post comment when engineering team is initializing"""
    message = """🔄 Initializing Engineering Team... / 正在初始化工程团队..."""
    return post_comment(work_item_id, message)

def post_workflow_started(work_item_id: int) -> Dict[str, Any]:
    """Post comment when workflow starts"""
    message = """🔄 CrewAI workflow started / CrewAI 工作流程已启动 """
    return post_comment(work_item_id, message)


def update_status(work_item_id: int, new_state: str) -> Dict[str, Any]:
    """
    Update the status of an Azure DevOps work item
    """
    url = f"{BASE_URL}/wit/workitems/{work_item_id}?api-version=7.0"
    headers = {"Content-Type": "application/json-patch+json"}
    payload = [
        {
            "op": "replace",
            "path": "/fields/System.State",
            "value": new_state
        }
    ]
    
    logger.info(f"Updating work item {work_item_id} status to: {new_state}")
    
    return _make_request("patch", url, headers=headers, json=payload)

def get_work_item(work_item_id: int) -> Optional[Dict[str, Any]]:
    """
    Get details of a specific work item
    """
    url = f"{BASE_URL}/wit/workitems/{work_item_id}?api-version=7.0"
    
    try:
        return _make_request("get", url)
    except ADOClientError:
        logger.warning(f"Work item {work_item_id} not found or access denied")
        return None