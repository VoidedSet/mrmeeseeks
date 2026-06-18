import streamlit as st
import os
import json
import time
import subprocess
from datetime import datetime
import psutil

# Set Page Config
st.set_page_config(
    page_title="Mr Meeseeks Profiler",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styles
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Title styling */
    .title-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 12px;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C6FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        color: #8b949e;
        margin-left: auto;
        font-size: 0.9rem;
    }

    /* Chat layout styling */
    .chat-card {
        background-color: #0d1117;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #30363d;
    }
    .user-bubble {
        background-color: #1f6feb;
        color: white;
        padding: 10px 14px;
        border-radius: 18px 18px 2px 18px;
        margin-bottom: 12px;
        max-width: 80%;
        margin-left: auto;
        text-align: left;
        font-size: 14.5px;
        font-weight: 400;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .agent-bubble {
        background-color: #238636;
        color: white;
        padding: 10px 14px;
        border-radius: 18px 18px 18px 2px;
        margin-top: 12px;
        max-width: 80%;
        margin-right: auto;
        text-align: left;
        font-size: 14.5px;
        font-weight: 400;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    /* Trace details */
    .trace-summary {
        cursor: pointer;
        padding: 8px;
        background-color: #21262d;
        border-radius: 6px;
        border: 1px solid #30363d;
        font-size: 13px;
        font-weight: 600;
        color: #c9d1d9;
        margin: 8px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .trace-details-content {
        background-color: #161b22;
        border-radius: 6px;
        border: 1px solid #21262d;
        padding: 12px;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .trace-line {
        margin: 6px 0;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 13px;
        line-height: 1.4;
    }
    .time-badge {
        background-color: #30363d;
        color: #8b949e;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        flex-shrink: 0;
        border: 1px solid #444c56;
    }
    .amber-text {
        color: #ffc107;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .teal-text {
        color: #00e5ff;
        font-weight: 600;
    }
    .purple-text {
        color: #e040fb;
        font-weight: 600;
    }
    .grey-text {
        color: #8b949e;
    }
    
    /* Code and result previews */
    .result-preview {
        background-color: #0d1117;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #21262d;
        color: #8b949e;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        white-space: pre-wrap;
        word-break: break-all;
        margin-top: 4px;
        max-height: 120px;
        overflow-y: auto;
    }

    /* Tab colors and alerts */
    .alert-banner {
        padding: 12px;
        border-radius: 6px;
        background-color: #8f1f1f20;
        border: 1px solid #f85149;
        color: #f85149;
        font-weight: bold;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Session States
if "sentinel_clear_time" not in st.session_state:
    st.session_state.sentinel_clear_time = 0.0

if "trace_expanded" not in st.session_state:
    st.session_state.trace_expanded = {}

# Constants
PROFILER_FILE = "/tmp/meeseeks_profiler.jsonl"
STORE_DIR = "/home/kshayik/Projects/mr-meeseeks/memory/store"

# Helper: Read Profiler Log
def read_profiler_log():
    if not os.path.exists(PROFILER_FILE):
        return []
    events = []
    try:
        with open(PROFILER_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        st.sidebar.error(f"Error reading log: {e}")
    return events

# Helper: Parse turns from events
def group_events_into_turns(events):
    turns = []
    if not events:
        return turns
        
    current_turn = {
        "prompt": "System Initialization",
        "start_ts": events[0]["ts"],
        "events": [],
        "agent_response": None
    }
    
    for event in events:
        if event["type"] == "user_input":
            # Push last turn if it had activity
            if current_turn["events"] or current_turn["prompt"] != "System Initialization":
                turns.append(current_turn)
            current_turn = {
                "prompt": event["prompt"],
                "start_ts": event["ts"],
                "events": [],
                "agent_response": None
            }
        elif event["type"] == "agent_response":
            current_turn["agent_response"] = event.get("speech", "")
            current_turn["events"].append(event)
        else:
            current_turn["events"].append(event)
            
    turns.append(current_turn)
    return turns

# Helper: Find mr-meeseeks processes
def get_meeseeks_processes():
    pids = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        try:
            cwd = p.info.get('cwd')
            cmdline = p.info.get('cmdline') or []
            # Match mr-meeseeks directory or main.py in cmdline
            if cwd == "/home/kshayik/Projects/mr-meeseeks" or any("main.py" in arg for arg in cmdline):
                pids.append(p)
                for child in p.children(recursive=True):
                    pids.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    # Deduplicate processes
    unique_procs = {}
    for p in pids:
        try:
            unique_procs[p.pid] = p
        except Exception:
            pass
    return list(unique_procs.values())

# Helper: Check swap warning
def check_swap_warning(processes):
    for p in processes:
        try:
            mem = p.memory_info()
            # If virtual memory size is more than 2x Resident Set Size
            if mem.vms > mem.rss * 2:
                return True, f"⚠️ process '{p.name()}' (PID {p.pid}) is hitting swap space! RSS={mem.rss//(1024*1024)}MB, VMS={mem.vms//(1024*1024)}MB."
        except Exception:
            pass
    return False, ""

# Helper: Get nvidia-smi GPU data
def get_gpu_data(target_pids):
    gpu_status = None
    target_gpu_apps = []
    
    # 1. Get global GPU utilization
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.split(",")]
        if len(parts) >= 6:
            gpu_status = {
                "name": parts[0],
                "util_gpu": int(parts[1]),
                "util_mem": int(parts[2]),
                "total": int(parts[3]),
                "free": int(parts[4]),
                "used": int(parts[5])
            }
    except Exception:
        pass
        
    # 2. Get processes running on GPU
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        lines = res.stdout.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid = int(parts[0])
                if pid in target_pids:
                    target_gpu_apps.append({
                        "pid": pid,
                        "name": parts[1],
                        "vram": int(parts[2])
                    })
    except Exception:
        pass
        
    return gpu_status, target_gpu_apps

# Helper: Load SuperMemory Store
def load_supermemory_store():
    data = []
    if os.path.exists(STORE_DIR):
        for filename in os.listdir(STORE_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(STORE_DIR, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    data.append({
                        "file": filename,
                        "content": content,
                        "mtime": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
                except Exception:
                    pass
    return sorted(data, key=lambda x: x["file"])


# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h2 style='text-align: center; color: #0072FF;'>🛠️ Profiler Settings</h2>", unsafe_allow_html=True)

# Auto refresh control
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (1s)", value=True)
refresh_interval = 1.0

# Clear Log Button
if st.sidebar.button("Clear Session Profiler File"):
    if os.path.exists(PROFILER_FILE):
        try:
            os.remove(PROFILER_FILE)
            st.sidebar.success("Cleared profiler log file!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing: {e}")
    else:
        st.sidebar.info("Log file is already empty.")

# Process Discovery Status
meeseeks_procs = get_meeseeks_processes()
meeseeks_pids = [p.pid for p in meeseeks_procs]
st.sidebar.markdown(f"**Discovered PIDs**: `{len(meeseeks_pids)}` active")
for p in meeseeks_procs[:5]:
    try:
        st.sidebar.markdown(f"- `{p.pid}`: **{p.name()}**")
    except Exception:
        pass
if len(meeseeks_procs) > 5:
    st.sidebar.markdown(f"*and {len(meeseeks_procs) - 5} more children*")

# Swap Warning Detection
has_swap_warning, swap_msg = check_swap_warning(meeseeks_procs)


# --- MAIN HEADER ---
st.markdown(f"""
<div class="title-container">
    <div class="title-text">MR MEESEEKS PROFILER</div>
    <div class="subtitle-text">Session Live | Local Server Port 6767</div>
</div>
""", unsafe_allow_html=True)

if has_swap_warning:
    st.markdown(f'<div class="alert-banner">{swap_msg}</div>', unsafe_allow_html=True)


# --- COLUMNS LAYOUT ---
col_left, col_right = st.columns([3, 2])

# Load log events
log_events = read_profiler_log()


# ==========================================
# PANE 1: CHAT & TRACE (LEFT COLUMN)
# ==========================================
with col_left:
    st.markdown("### 💬 Conversation & ReAct Trace")
    
    turns = group_events_into_turns(log_events)
    
    if not turns:
        st.info("No traces found. Send a prompt to Mr Meeseeks to populate this view.")
    else:
        for idx, turn in enumerate(turns):
            st.markdown(f'<div class="chat-card">', unsafe_allow_html=True)
            
            # 1. User Prompt bubble
            if turn["prompt"] == "System Initialization":
                st.markdown(f'<div style="color: #8b949e; text-align: center; margin-bottom: 12px; font-weight: 600; font-size: 13px;">⚙️ {turn["prompt"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-bubble">🧑 <b>User:</b> {turn["prompt"]}</div>', unsafe_allow_html=True)
            
            # 2. Collapsible Insider Processing details
            # We can manage expand state in session_state or just default to expanded for latest turn
            is_latest = (idx == len(turns) - 1)
            expand_key = f"expand_turn_{idx}"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = is_latest
                
            expand_clicked = st.checkbox(f"🔍 Show Trace Details (Turn #{idx})", value=st.session_state[expand_key], key=f"cb_{idx}")
            st.session_state[expand_key] = expand_clicked
            
            if expand_clicked:
                st.markdown('<div class="trace-details-content">', unsafe_allow_html=True)
                
                # Iterate events and display them
                for ev in turn["events"]:
                    elapsed_ms = int((ev["ts"] - turn["start_ts"]) * 1000)
                    time_badge = f'<span class="time-badge">+{elapsed_ms}ms</span>'
                    
                    if ev["type"] == "state_change":
                        st.markdown(f'<div class="trace-line">{time_badge} 🔄 State: <span class="teal-text">{ev.get("state")}</span></div>', unsafe_allow_html=True)
                        
                    elif ev["type"] == "function_call":
                        status = ev.get("status", "start")
                        name = ev.get("name", "")
                        if status == "start":
                            st.markdown(f'<div class="trace-line">{time_badge} 🔧 Calling <span class="amber-text">{name}()</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="trace-line">{time_badge} ✓ <span class="amber-text">{name}()</span> finished</div>', unsafe_allow_html=True)
                            
                    elif ev["type"] == "llm_start":
                        st.markdown(f'<div class="trace-line">{time_badge} ⏳ LLM Request initiated ({ev.get("provider")})</div>', unsafe_allow_html=True)
                        
                    elif ev["type"] == "llm_chunk":
                        # Summarized info to prevent cluttering
                        pass
                        
                    elif ev["type"] == "llm_end":
                        is_agentic = ev.get("is_agentic", False)
                        err = ev.get("error")
                        if err:
                            st.markdown(f'<div class="trace-line">{time_badge} 🛑 LLM Error: <span style="color: #ff7b72;">{err}</span></div>', unsafe_allow_html=True)
                        elif is_agentic:
                            tc = ev.get("tool_call", {})
                            st.markdown(f'<div class="trace-line">{time_badge} 🤖 LLM complete (Agentic route) → Tool call: <span class="purple-text">{tc.get("tool")}</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="trace-line">{time_badge} 🤖 LLM complete (Conversational route)</div>', unsafe_allow_html=True)
                            
                    elif ev["type"] == "react_step":
                        st.markdown(f'<div class="trace-line">{time_badge} ⚡ ReAct Step {ev.get("step")}: <i>"{ev.get("thought")}"</i></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="padding-left: 20px; font-size:12.5px;" class="trace-line">└─ Action Plan: <span class="teal-text">{ev.get("tool")}</span> with args: <code style="color:#d49fff;">{json.dumps(ev.get("args"))}</code></div>', unsafe_allow_html=True)
                        
                    elif ev["type"] == "tool_dispatch":
                        st.markdown(f'<div class="trace-line">{time_badge} 🚀 Dispatching Tool: <span class="purple-text">{ev.get("tool")}</span></div>', unsafe_allow_html=True)
                        
                    elif ev["type"] == "tool_result":
                        res_str = ev.get("result", "")
                        st.markdown(f'<div class="trace-line">{time_badge} 📥 Tool result for <span class="purple-text">{ev.get("tool")}</span>:</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="margin-left: 30px;" class="result-preview">{res_str[:400]}...</div>', unsafe_allow_html=True)
                        
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. Agent speech response bubble
            if turn["agent_response"]:
                st.markdown(f'<div class="agent-bubble">🤖 <b>Meeseeks:</b> {turn["agent_response"]}</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# RIGHT COLUMN (GPU + CPU MONITORS)
# ==========================================
with col_right:
    # PANE 2: GPU MONITOR
    st.markdown("### 🎮 GPU Monitor")
    
    gpu_status, gpu_apps = get_gpu_data(meeseeks_pids)
    
    if gpu_status is None:
        st.info("NVIDIA GPU / nvidia-smi utility is not detected on the system.")
    else:
        st.markdown(f"**GPU Model**: `{gpu_status['name']}`")
        
        # VRAM Gauge / Util percentage
        vram_col, util_col = st.columns(2)
        vram_col.metric("VRAM Usage", f"{gpu_status['used']} / {gpu_status['total']} MB", f"{int(gpu_status['used'] / gpu_status['total'] * 100)}%")
        util_col.metric("GPU Utilization", f"{gpu_status['util_gpu']}%", f"Memory: {gpu_status['util_mem']}%")
        
        if not gpu_apps:
            st.write("No active Mr Meeseeks processes on GPU.")
        else:
            st.markdown("**Processes using VRAM:**")
            # Build list
            for app in gpu_apps:
                st.markdown(f"- **PID {app['pid']}** ({app['name']}): `{app['vram']} MB` VRAM")
                
    st.markdown("---")
    
    # PANE 3: CPU MONITOR
    st.markdown("### 💻 CPU & RAM Monitor")
    
    if not meeseeks_procs:
        st.info("No active Python process tree found for Mr Meeseeks. Ensure the main agent is running.")
    else:
        cpu_total = 0.0
        ram_total_rss = 0.0
        ram_total_vms = 0.0
        
        proc_rows = []
        for p in meeseeks_procs:
            try:
                # Need to run CPU percent in non-blocking way
                cpu_p = p.cpu_percent(interval=None) or 0.0
                mem = p.memory_info()
                cpu_total += cpu_p
                ram_total_rss += mem.rss
                ram_total_vms += mem.vms
                proc_rows.append({
                    "PID": p.pid,
                    "Name": p.name(),
                    "CPU%": f"{cpu_p:.1f}%",
                    "RSS (RAM)": f"{mem.rss // (1024 * 1024)} MB",
                    "VMS (Virtual)": f"{mem.vms // (1024 * 1024)} MB"
                })
            except Exception:
                pass
                
        # Metrics summary
        cpu_col, ram_col = st.columns(2)
        cpu_col.metric("Total CPU Usage", f"{cpu_total:.1f}%")
        ram_col.metric("Total RAM (RSS)", f"{ram_total_rss // (1024*1024)} MB", f"VMS: {ram_total_vms // (1024*1024)} MB")
        
        # Display process list
        st.dataframe(proc_rows, use_container_width=True)


st.markdown("---")

# ==========================================
# BOTTOM REGION (SUPERMEMORY + KERNEL PACKETS)
# ==========================================
col_bot1, col_bot2 = st.columns(2)

# PANE 4: SUPERMEMORY STORAGE (BOTTOM LEFT)
with col_bot1:
    st.markdown("### 📂 SuperMemory Store Preview")
    
    stores = load_supermemory_store()
    
    if not stores:
        st.info("No local JSON key-value store files found in memory/store/.")
    else:
        # File selector dropdown
        file_names = [s["file"] for s in stores]
        selected_file = st.selectbox("Select Key-Value store file:", file_names)
        
        # Find matches
        matched_store = next(s for s in stores if s["file"] == selected_file)
        st.markdown(f"**Last modified**: `{matched_store['mtime']}`")
        
        content = matched_store["content"]
        if not content:
            st.write("Store is empty.")
        else:
            if isinstance(content, dict):
                # Expandable lists for keys
                for key, val in content.items():
                    val_preview = str(val)[:120] + ("..." if len(str(val)) > 120 else "")
                    with st.expander(f"🔑 `{key}`  —  {val_preview}"):
                        st.json(val)
            else:
                st.json(content)


# PANE 5: KERNEL PACKETS (BOTTOM RIGHT)
with col_bot2:
    st.markdown("### ⚡ Sentinel Kernel Packets")
    
    # Extract sentinel events
    sentinel_events = [ev for ev in log_events if ev["type"] == "sentinel_event"]
    
    # Filter by user clear time
    sentinel_events = [ev for ev in sentinel_events if ev["ts"] > st.session_state.sentinel_clear_time]
    
    # Filter controls
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    filter_query = col_ctrl1.text_input("Filter packets by text:", "")
    if col_ctrl2.button("Clear View", use_container_width=True):
        st.session_state.sentinel_clear_time = time.time()
        st.rerun()
        
    # Render scrolling table
    if not sentinel_events:
        st.info("No proactive kernel events logged in this session.")
    else:
        packet_rows = []
        for ev in reversed(sentinel_events):
            payload = ev.get("payload", {})
            collector = payload.get("collector", "generic")
            details = payload.get("details", {})
            event_type = payload.get("type", "unknown")
            event_msg = payload.get("event", "")
            
            # Simple string match filter
            raw_text = f"{collector} {event_type} {event_msg} {str(details)}".lower()
            if filter_query and filter_query.lower() not in raw_text:
                continue
                
            # Formatting timestamp
            dt = datetime.fromtimestamp(ev["ts"]).strftime('%H:%M:%S.%f')[:-3]
            
            # Color indicator by collector type
            color_emoji = "🟣"
            if "inotify" in collector or "file" in collector:
                color_emoji = "🔵"  # File System (inotify)
            elif "process" in collector:
                color_emoji = "🟢"  # Processes
            elif "network" in collector:
                color_emoji = "🟡"  # Network
            elif "device" in collector or "battery" in collector:
                color_emoji = "🟣"  # Device / battery
                
            packet_rows.append({
                "Time": dt,
                "Type": f"{color_emoji} {collector.upper()}",
                "Event": event_msg,
                "Details": json.dumps(details)
            })
            
        if not packet_rows:
            st.write("No matching packets found.")
        else:
            st.dataframe(packet_rows, use_container_width=True)


# Auto rerun trigger at the bottom
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
