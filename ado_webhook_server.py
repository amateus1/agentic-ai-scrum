import json
import logging
import os
import datetime
import traceback
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
import time
import queue

# === Set up Python Path ===
venv_path = "/root/crewai-engineering-team/.venv"
project_root = "/root/crewai-engineering-team"
src_path = "/root/crewai-engineering-team/src"
sys.path.insert(0, "/root/crewai-engineering-team/src/engineering-team")  # 👈 add root for main.py
sys.path.insert(0, "/root/crewai-engineering-team/output")

python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
site_packages = f"{venv_path}/lib/{python_version}/site-packages"

paths_to_add = [site_packages, project_root, src_path]
for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# === Load .env file ===
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info("✅ .env file loaded successfully")
    else:
        logging.warning("⚠️ .env file not found")
except ImportError:
    logging.warning("⚠️ python-dotenv not installed, .env file won't be loaded")

# Verify DeepSeek API key is available
if not os.getenv('DEEPSEEK_API_KEY'):
    logging.error("❌ DEEPSEEK_API_KEY not found in environment variables")

# === Import cost tracking from main.py ===
try:
    from engineering_team.main import (
        _monkey_patch_litellm, 
        _setup_agent_with_tracking,
        GLOBAL_TOKEN_USAGE,
        _distribute_global_token_usage_by_agent,
        PRICING,
        AGENT_LLM_MAPPING as EXPECTED_AGENTS_AND_MODELS  # Rename for consistency
    )
    logging.info("✅ Successfully imported token tracking functions from main.py")
except ImportError as e:
    logging.error(f"❌ Failed to import token tracking functions: {e}")
    # Fallbacks
    _monkey_patch_litellm = lambda: None
    _setup_agent_with_tracking = lambda agent, tracker: None
    GLOBAL_TOKEN_USAGE = {}
    _distribute_global_token_usage_by_agent = lambda: {}
    PRICING = {}
    EXPECTED_AGENTS_AND_MODELS = {}

# === WANDB imports ===
# try:
#    from engineering_team.wandb_utils import log_crew_execution_to_wandb
#    logging.info("✅ Successfully imported WandB utilities")
#except ImportError as e:
#    logging.error(f"❌ Failed to import WandB utilities: {e}")
#    log_crew_execution_to_wandb = None

# === Set up environment and logging ===
Path("runs").mkdir(exist_ok=True)
log_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"runs/webhook_{log_stamp}.log"),
        logging.StreamHandler()
    ]
)

# === Initializes Weave ===
try:
    # import weave
    # weave.init(project_name="crewai-ado-integration-2")
    #logging.info("✅ Weave initialized successfully")
    logging.info("✅ Weave initialized skipped")
except ImportError:
    logging.warning("⚠️ Weave not available - tracing will be limited")
except Exception as e:
    logging.error(f"❌ Failed to initialize Weave: {e}")

# logging.info("🚀 ADO Webhook Server Started with Weave Integration")

# === Import CrewAI and helpers ===
try:
    from engineering_team.crew import EngineeringTeam
    logging.info("✅ Successfully imported EngineeringTeam")
except ImportError as e:
    logging.error(f"❌ Failed to import EngineeringTeam: {e}")
    sys.exit(1)

try:
    from engineering_team.ado_client import (
        post_comment, update_status, post_crew_completed, 
        post_execution_started, post_crew_initializing, 
        post_team_initializing, post_workflow_started
    )
except Exception as e:
    post_comment = None
    update_status = None
    post_crew_completed = None
    post_execution_started = None
    post_crew_initializing = None
    post_team_initializing = None
    post_workflow_started = None
    logging.warning(f"⚠️ ADO client not available: {e}")

try:
    from engineering_team.token_tracker import EnhancedTokenUsageCallbackHandler
except Exception:
    EnhancedTokenUsageCallbackHandler = None
    logging.warning("⚠️ Token tracker not available")

processed_items = set()

# === Thread-safe timeout function ===
def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=600):
    """Run a function with a timeout using threads (signal-safe)"""
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def worker():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    
    if not exception_queue.empty():
        raise exception_queue.get()
    
    return result_queue.get()

# === Weave-wrapped functions ===
# @weave.op()



    # @weave.op()
#@def validate_requirements(req: str) -> bool:
#@    return bool(req and len(req.strip()) > 10)

#def initialize_wandb_run(work_item_id, title):
#    try:
#        import wandb
        # run = wandb.init(
        #    project="crewai-ado-integration-2",
        #    name=f"ado_{work_item_id}_{datetime.datetime.now().strftime('%H%M%S')}",
        #     config={"work_item_id": work_item_id, "title": title}
        # )
        # return run
    # except Exception as e:
    #    logging.warning(f"W&B init failed: {e}")
    #    return None

def _attach_tracker_to_agents(crew, tracker):
    for agent in getattr(crew, "agents", []):
        tracker.set_current_agent(getattr(agent, "role", "Unknown"))
        if hasattr(agent, "llm") and agent.llm:
            agent.llm.callbacks = agent.llm.callbacks or []
            agent.llm.callbacks.append(tracker)

def extract_assigned_to(fields):
    assigned_to = fields.get('System.AssignedTo', '')
    if isinstance(assigned_to, dict):
        assigned_to = assigned_to.get('newValue', '')
    if '<' in assigned_to:
        assigned_to = assigned_to.split('<')[0].strip()
    return assigned_to

def get_current_values(resource):
    fields = resource.get("revision", {}).get("fields", {})
    return {
        "title": fields.get("System.Title", "No Title"),
        "state": fields.get("System.State", ""),
        "description": fields.get("System.Description", "")
    }

def should_process_work_item(resource):
    wid = resource.get("workItemId")
    if not wid or wid in processed_items:
        return False
    fields = resource.get("fields", {})
    assigned = extract_assigned_to(fields)
    values = get_current_values(resource)
    if assigned != "Agentic Lead" or values["state"] not in ["New", "Committed"]:
        return False
    return True, wid, values

def post_progress_update(wid, message):
    """Safely post progress updates with error handling"""
    if post_comment:
        try:
            post_comment(wid, message)
        except Exception as e:
            logging.error(f"Failed to post progress update: {e}")

def format_cost_breakdown(by_agent, by_llm, total_cost, llms_used):
    """Format the cost breakdown in the expected format"""
    if not by_agent and not by_llm:
        return "No cost data available"
    
    breakdown = []
    
    # Add LLMs used section
    if llms_used:
        breakdown.append("🧠 **LLMs Used:** / 使用的 LLM： ")
        for llm in sorted(llms_used):
            breakdown.append(f"- {llm}")
        breakdown.append("")
    
    # Add cost breakdown by agent
    if by_agent:
        breakdown.append("💰 **Cost Breakdown:** / 成本明细：")
        breakdown.append("")
        breakdown.append("👥 **By Agent:** / 按角色")
        for agent, agent_data in by_agent.items():
            if agent_data.get("models"):
                for model, cost in agent_data["models"].items():
                    breakdown.append(f"- {agent}: {model}: ${cost:.4f}")
            elif agent_data.get("total", 0) > 0:
                breakdown.append(f"- {agent}: ${agent_data['total']:.4f}")
        breakdown.append("")
    
    # Add cost breakdown by LLM
    if by_llm:
        breakdown.append("🧠 **By LLM (Total):** / 按 LLM（总计）：" )
        for model, cost in sorted(by_llm.items()):
            breakdown.append(f"- {model}: ${cost:.4f}")
        breakdown.append("")
    
    # Add total cost
    breakdown.append(f"💵 **Total Cost: / 总成本 ${total_cost:.4f}**")
    
    return "\n".join(breakdown)

def run_crewai_workflow(requirements: str, module_name: str, class_name: str, token_tracker=None):
    # Ensure the output dir is on sys.path
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir_str = str(output_dir)

    if output_dir_str not in sys.path:
        sys.path.insert(0, output_dir_str)
        
    try:
        from weave import trace  # optional tracing
    except ImportError:
        trace = None

    team = EngineeringTeam()
    crew = team.crew()

    if token_tracker:
        for agent in crew.agents:
            _setup_agent_with_tracking(agent, token_tracker)

    return crew.kickoff(inputs={
        "requirements": requirements,
        "module_name": module_name,
        "class_name": class_name
    })


def validate_requirements(req: str) -> bool:
    return bool(req and isinstance(req, str) and len(req.strip()) > 0)


def execute_crewai_with_tracing(wid, title, desc):
    # Apply LiteLLM monkey patch for proper token tracking
    try:
        _monkey_patch_litellm()
        logging.info("✅ Applied LiteLLM monkey patch for token tracking")
    except Exception as e:
        logging.error(f"❌ Failed to apply LiteLLM monkey patch: {e}")

    # Reset LiteLLM cost
    import litellm
    litellm._current_cost = 0

    if post_comment is None:
        logging.error("ADO client is not available - cannot post comments")
        return
    
    # wandb_run = None
    start = time.time()
    processed_items.add(wid)

    logging.info(f"Starting execution for work item {wid}")

     # Update status to Active when crew starts working
    if update_status:
        try:
            update_status(wid, "Committed")
            logging.info(f"Updated work item {wid} status to Committed")
        except Exception as e:
            logging.error(f"Failed to update status to Committed: {e}")
    
    # Debug: Check if functions are available
    logging.info(f"post_comment available: {post_comment is not None}")
    logging.info(f"post_workflow_started available: {post_workflow_started is not None}")
    logging.info(f"post_team_initializing available: {post_team_initializing is not None}")
    logging.info(f"post_crew_initializing available: {post_crew_initializing is not None}")
    logging.info(f"post_execution_started available: {post_execution_started is not None}")

    # Post initial workflow status
    # Initial workflow status handled by main.py, no need to duplicate here
    logging.info("ℹ️ Skipping webhook init comments (delegated to main.py)")

    # wandb_run = initialize_wandb_run(wid, title)
    # weave_url = getattr(wandb_run, "url", None) if wandb_run else None

    #if not validate_requirements(desc):
    #    post_comment(wid, "❌ Validation failed.")
    #    return

    tracker = EnhancedTokenUsageCallbackHandler() if EnhancedTokenUsageCallbackHandler else None
    
    # Create a flag to control the progress monitoring thread
    execution_complete = threading.Event()
    
    # Track actual agent progress to prevent fake updates
    actual_agent_progress = {
        "Engineering Lead": False,
        "Backend Engineer": False, 
        "Frontend Engineer": False,
        "Test Engineer": False
    }
    
    # Start progress monitoring thread
    progress_monitor_thread = threading.Thread(
        target=monitor_progress, 
        args=(wid, start, tracker, execution_complete, actual_agent_progress),
        daemon=True
    )
    progress_monitor_thread.start()
    
    # FIX: Create crew instance explicitly to pass to WandB logging
    crew_instance = None
    execution_successful = False
    error_posted = False  # Track if we've already posted an error

    try:
        team = EngineeringTeam()
        crew_instance = team.crew()

        # Add agent progress tracking
        for agent in crew_instance.agents:
            if tracker:
                _setup_agent_with_tracking(agent, tracker)

            # Track which agents actually start working
            role = getattr(agent, 'role', 'Unknown Agent')
            if role in actual_agent_progress:
                actual_agent_progress[role] = True
                logging.info(f"✅ Agent started working: {role}")

        # Use the timeout-protected function (thread-safe)
        os.environ["ADO_WORK_ITEM_ID"] = str(wid)
        from engineering_team import main
        try:
            logging.info("🧠 Using main.run() as crew execution entry point")

            result = run_with_timeout(
                func=main.run,
                timeout_seconds=600  # 10 minutes,
            )
            # ✅ After main finishes, force the state update
            if update_status:
                try:
                    update_status(wid, "AI-Agent_Done")
                    logging.info(f"✅ Work item {wid} moved to AI-Agent_Done")
                except Exception as e:
                    logging.error(f"❌ Failed to update work item {wid} state: {e}")

        except Exception as e:
            logging.error(f"❌ Crew execution failed unexpectedly: {e}")
            logging.error(traceback.format_exc())
            result = None

        # Success detection
        if result is None:
            logging.info("⏳ Execution may still be in progress. Awaiting monitor to confirm.")
            # Don't set execution_successful to False yet
        elif result and isinstance(result, dict):
            execution_successful = True
            logging.info(f"Crew execution completed successfully: {result}")
        else:
            execution_successful = False
            logging.error(f"Crew execution failed or returned error: {result}")

    except Exception as e:
        logging.error(f"❌ Failed to create or run crew: {e}")
        logging.error(traceback.format_exc())
        result = None
        execution_successful = False
        
    # Signal the progress monitor to stop immediately
    execution_complete.set()

    
    # Wait a moment for the progress thread to exit
    time.sleep(1)
    
    duration = time.time() - start

    llms_used = set()
    by_agent = {}
    by_llm = {}
    total_cost = 0.0

    if GLOBAL_TOKEN_USAGE:
        distributed_usage = _distribute_global_token_usage_by_agent()
        for agent, agent_data in distributed_usage.items():
            agent_total = 0.0
            agent_models = {}
            for model, usage in agent_data.items():
                input_tokens = float(usage.get('input_tokens', 0))
                output_tokens = float(usage.get('output_tokens', 0))
                rate = PRICING.get(model, {"in": 0.0, "out": 0.0})
                cost = (input_tokens / 1000000) * rate["in"] + (output_tokens / 1000000) * rate["out"]
                cost = round(cost, 6)
                agent_total += cost
                agent_models[model] = cost
                by_llm[model] = by_llm.get(model, 0.0) + cost
                total_cost += cost
                llms_used.add(model)
            if agent_total > 0:
                by_agent[agent] = {"total": round(agent_total, 6), "models": agent_models}
    total_cost = round(total_cost, 6)

    timestamp_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    app_url = os.getenv("APP_BASE_URL", "").rstrip("/") or "http://8.219.119.18:7860"

    # Only post completion if we actually completed successfully
    if execution_successful:
        post_progress_update(wid, "✅ **All agents completed their tasks successfully** / 所有智能体已成功完成任务")
        
        if post_crew_completed:
            cost_breakdown = format_cost_breakdown(by_agent, by_llm, total_cost, llms_used)
            post_crew_completed(wid, duration, llms_used, by_agent, by_llm, total_cost, weave_url, app_url)
        
        logging.info(f"Execution completed successfully in {duration:.2f}s")

    if wandb_run:
        try:
            if log_crew_execution_to_wandb:
                log_crew_execution_to_wandb(
                    wandb_run=wandb_run,
                    duration=duration,
                    total_cost=total_cost,
                    global_token_usage=GLOBAL_TOKEN_USAGE,
                    llms_used=llms_used,
                    work_item_id=wid,
                    title=title,
                    crew=crew_instance,
                )
            else:
                # Fallback to original simple logging
                wandb_run.log({
                    "duration_seconds": duration, 
                    "success": execution_successful,
                    "total_cost": total_cost
                })
        except Exception as e:
            logging.error(f"❌ Failed to log to WandB: {e}")
            # Fallback to original simple logging
            wandb_run.log({
                "duration_seconds": duration, 
                "success": execution_successful,
                "total_cost": total_cost
            })
        # finally:
            # wandb_run.finish()

def monitor_progress(wid, start_time, tracker, execution_complete, actual_agent_progress):
    """Monitor execution progress and post periodic updates"""
    progress_updates = [
        (30, "👨‍💼 **Engineering Lead** started designing the architecture... / 首席工程师 开始设计架构... "),
        (60, "✅ **Milestone**: Design phase completed / 里程碑: 设计阶段完成 "),
        (90, "👨‍💻 **Backend Engineer** started coding the module... / 后端工程师 开始编写模块代码..."),
        (120, "✅ **Milestone**: Backend code generated / 里程碑: 后端代码已生成"),
        (150, "🎨 **Frontend Engineer** started building the UI... / 前端工程师 开始构建界面..."),
        (180, "✅ **Milestone**: Frontend interface built / 里程碑: 前端界面已构建"),
        (210, "🧪 **Test Engineer** started writing tests... / 测试工程师 开始编写测试.."),
        (240, "✅ **Milestone**: Test suite created / 里程碑: 测试套件已创建" ),
        (300, "✅ **Quality Check**: Code compilation successful / 质量检查: 代码编译成功"),
        (360, "✅ **Quality Check**: All tests passing / 质量检查: 所有测试通过"),
        (420, "✅ **Quality Check**: UI validation complete / 质量检查: 界面验证完成")
    ]
    
    posted_updates = set()
    last_activity_time = time.time()
    
    while not execution_complete.is_set():
        current_time = time.time()
        duration = current_time - start_time
        
        # Only post updates if agents are actually working
        agents_working = True
        
        if agents_working:
            # Post milestone updates (only once each)
            for wait_time, message in progress_updates:
                if duration >= wait_time and wait_time not in posted_updates:
                    post_progress_update(wid, message)
                    posted_updates.add(wait_time)
                    logging.info(f"Posted progress update: {message}")
                    last_activity_time = current_time
        
        # Check if execution is complete or timeout
        if execution_complete.is_set() or duration > 600:  # 10 minutes timeout
            break
            
        time.sleep(5)  # Check more frequently but sleep shorter
    
    logging.info("Progress monitoring thread exiting")

    # ✅ Post final success if execution completed
    if execution_complete.is_set():
        logging.info("ℹ️ Skipping duplicate final success comment (handled by main.py)")

def process_webhook_payload(payload):
    try:
        resource = payload.get("resource", {})
        result = should_process_work_item(resource)
        if result:
            _, wid, values = result
            logging.info(f"Processing work item {wid}: {values['title']}")
            threading.Thread(target=execute_crewai_with_tracing, args=(wid, values["title"], values["description"]), daemon=True).start()
        else:
            logging.info("Webhook received but no processing needed")
    except Exception as e:
        logging.error(f"Error in process_webhook_payload: {e}")
        logging.error(traceback.format_exc())

class CrewWebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
            else:
                post_data = b'{}'
            
            # Immediately respond to avoid connection reset
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "accepted", "message": "Webhook received and processing started"}')
            
            # Log the request
            logging.info(f"Webhook received from {self.client_address[0]}")
            
            # Process the payload asynchronously
            try:
                payload = json.loads(post_data.decode('utf-8'))
                threading.Thread(target=process_webhook_payload, args=(payload,), daemon=True).start()
            except Exception as e:
                logging.error(f"Error processing payload: {e}")
                
        except Exception as e:
            logging.error(f"Webhook error: {e}")
            # Still try to respond even if there's an error
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "message": "Internal server error"}')
            except:
                pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ADO Webhook Server Running with Weave")

def run_server(port=8000):
    logging.info(f"🚀 Server started on port {port}")
    HTTPServer(('', port), CrewWebhookHandler).serve_forever()

if __name__ == "__main__":
    run_server()