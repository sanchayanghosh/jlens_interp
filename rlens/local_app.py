import os
import json
import time
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pydantic
import uvicorn

import traceback

# Directory on your local PC where readouts are permanently stored
STORAGE_DIR = os.path.abspath("./saved_runs")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="Neuronpedia-Style Local R-Lens Viewer")

class RunRequest(pydantic.BaseModel):
    prompt: str
    max_new_tokens: int = 15

# Add this schema near the top
class MultiTurnRequest(pydantic.BaseModel):
    user_turns: list[str]
    max_new_tokens: int = 50



@app.post("/api/run_multiturn")
def trigger_multiturn_run(req: MultiTurnRequest):
    print(f"Sending Multi-turn request with {len(req.user_turns)} turns to Modal...")
    try:
        # The most robust way to call a class method in the modern Modal SDK
        engine_cls = modal.Cls.from_name("qwen-lens-backend", "RLensEngine")
        
        engine = engine_cls()
        
        payload = engine.run_multiturn_trajectory.remote(
            user_turns=req.user_turns,
            max_new_tokens=req.max_new_tokens
        )

    except Exception as e:
        # This will print the full, detailed error in the terminal running local_app.py
        print("\n--- FULL MODAL ERROR ---")
        traceback.print_exc()
        print("------------------------\n")
        
        # Use repr(e) instead of str(e) to prevent empty error strings
        raise HTTPException(status_code=500, detail=f"Modal Execution Error: {repr(e)}")

    # Add metadata and save locally
    run_id = f"multi_run_{int(time.time())}"
    save_data = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": req.user_turns[0] + " (Multi-turn)",
        **payload
    }

    file_path = os.path.join(STORAGE_DIR, f"{run_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    
    return JSONResponse(content=save_data)



@app.post("/api/run")
def trigger_modal_run(req: RunRequest):
    """Triggers the remote Modal GPU, fetches the full trajectory, and writes it to disk."""
    print(f"Calling Modal GPU backend for prompt: {req.prompt}")
    try:
        # Lookup the deployed Modal Function
        engine_cls = modal.Cls.lookup("qwen-lens-backend", "RLensEngine")
        engine = engine_cls()
        payload = engine.run_lens_trajectory.remote(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modal Execution Error: {str(e)}")

    # Add metadata for local storage
    run_id = f"run_{int(time.time())}"
    save_data = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": req.prompt,
        **payload
    }

    # Save permanently to local disk
    file_path = os.path.join(STORAGE_DIR, f"{run_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved run locally to {file_path}")

    return JSONResponse(content=save_data)

@app.get("/api/runs")
def list_saved_runs():
    """Lists all runs saved on the local PC."""
    runs = []
    for fname in sorted(os.listdir(STORAGE_DIR), reverse=True):
        if fname.endswith(".json"):
            path = os.path.join(STORAGE_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs.append({
                        "run_id": data.get("run_id", fname.replace(".json", "")),
                        "timestamp": data.get("timestamp", "Unknown"),
                        "prompt": data.get("prompt", "")[:60] + "...",
                        "answer": data.get("answer", "")
                    })
            except Exception:
                continue
    return JSONResponse(content=runs)

@app.get("/api/runs/{run_id}")
def get_saved_run(run_id: str):
    """Loads a specific run directly from the local disk (no GPU needed)."""
    file_path = os.path.join(STORAGE_DIR, f"{run_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Run not found on local disk")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/", response_class=HTMLResponse)
def serve_gui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Qwen R-Lens / Template Lens Local Explorer</title>
    <style>
        :root {
            --bg-main: #0d1117; --bg-card: #161b22; --border: #30363d;
            --text-main: #e6edf3; --text-muted: #8b949e; --accent-rlens: #8957e5;
            --accent-template: #2f81f7;
        }
        * { box-sizing: border-box; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
        body { background-color: var(--bg-main); color: var(--text-main); margin: 0; padding: 20px; display: flex; height: 100vh; overflow: hidden; }
        
        /* Layout */
        #history-sidebar { width: 280px; border-right: 1px solid var(--border); padding-right: 16px; display: flex; flex-direction: column; }
        #main-content { flex: 1; padding-left: 20px; display: flex; flex-direction: column; overflow: hidden; }
        .history-list { flex: 1; overflow-y: auto; margin-top: 10px; }
        .history-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; padding: 8px; margin-bottom: 8px; cursor: pointer; font-size: 11px; }
        .history-item:hover { border-color: var(--accent-template); }
        
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 14px; margin-bottom: 12px; }
        textarea { width: 100%; height: 50px; background: #010409; color: white; border: 1px solid var(--border); padding: 8px; border-radius: 4px; }
        
        /* Control Bar */
        .controls { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
        button.btn-run { background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        button.btn-run:hover { background: #2ea043; }
        
        .view-toggle button { background: var(--bg-card); color: var(--text-muted); border: 1px solid var(--border); padding: 6px 12px; cursor: pointer; border-radius: 4px; font-size: 12px; }
        .view-toggle button.active { background: #30363d; color: #fff; font-weight: bold; border-color: #58a6ff; }
        
        /* Viewer Matrix */
        .viewer-container { display: flex; flex: 1; gap: 16px; overflow: hidden; }
        .matrix-box { flex: 1; overflow: auto; border: 1px solid var(--border); background: var(--bg-card); border-radius: 6px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid var(--border); padding: 5px 8px; text-align: center; font-size: 11px; white-space: nowrap; }
        th { background: #1f242c; position: sticky; top: 0; z-index: 2; }
        th.token-hdr { cursor: pointer; }
        th.token-hdr:hover { background: #30363d; outline: 1px solid #58a6ff; }
        .gen-token { border-bottom: 2px solid #2f81f7; }
        td.cell { cursor: crosshair; max-width: 140px; overflow: hidden; text-overflow: ellipsis; }
        
        .inspector-panel { width: 300px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 14px; overflow-y: auto; }
        
        /* Modal */
        .modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--bg-card); border: 1px solid var(--border); padding: 20px; border-radius: 8px; z-index: 100; box-shadow: 0 0 20px rgba(0,0,0,0.8); }
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 99; }
    </style>
</head>
<body>

<div id="history-sidebar">
    <h3 style="margin-top:0; font-size:14px;">💾 Saved Local Runs</h3>
    <div style="font-size:11px; color:var(--text-muted);">Stored on your PC (No GPU required)</div>
    <div class="history-list" id="historyList">Loading...</div>
</div>

<div id="main-content">
    <div class="card">
        <label style="font-size:12px; color:var(--text-muted); display:block; margin-bottom:4px;">Input Prompt (Run on Modal GPU):</label>
        <textarea id="promptInput">The capital of the country where the Eiffel Tower is located is</textarea>
        <div class="controls">
            <div>
                <button class="btn-run" id="runBtn" onclick="runNewInference()">🚀 Execute & Save Locally</button>
                <span id="statusTxt" style="font-size:12px; color:var(--text-muted); margin-left:10px;"></span>
            </div>
            <div class="view-toggle">
                <button id="btnTmpl" class="active" onclick="switchView('template')">Template Lens (Phrases)</button>
                <button id="btnTok" onclick="switchView('token')">R-Lens (Tokens)</button>
            </div>
        </div>
        <div id="answerOut" style="margin-top:8px; font-size:12px; color:#58a6ff; font-weight:bold;"></div>
    </div>

    <div class="viewer-container">
        <div class="matrix-box" id="matrixBox">
            <div style="text-align:center; padding:50px; color:var(--text-muted);">Select a run from history or click Execute.</div>
        </div>
        <div class="inspector-panel" id="inspector">
            <h4 style="margin-top:0; font-size:13px;">Neuronpedia Readout</h4>
            <div id="inspectorBody" style="color:var(--text-muted); font-size:11px;">Hover over any cell to see the top-k vocabulary or phrase distribution.</div>
        </div>
    </div>
</div>

<!-- Token Swap / Steering Modal -->
<div class="overlay" id="overlay" onclick="closeSwapModal()"></div>
<div class="modal" id="swapModal">
    <h4 style="margin-top:0;">Steer / Counterfactual Swap</h4>
    <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">Editing token at Pos <span id="swapPos"></span></div>
    <input type="text" id="swapInput" style="width:100%; background:#010409; color:#fff; border:1px solid var(--border); padding:8px; margin-bottom:12px;" />
    <button class="btn-run" onclick="executeTokenSwap()">Run Counterfactual on GPU</button>
</div>

<script>
let currentData = null;
let currentMode = 'template'; // 'template' or 'token'

async function loadHistory() {
    const res = await fetch('/api/runs');
    const runs = await res.json();
    const list = document.getElementById('historyList');
    list.innerHTML = '';
    runs.forEach(r => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `<strong>${r.run_id}</strong><br><span style="color:#8b949e;">${r.timestamp}</span><br>${escapeHtml(r.prompt)}`;
        item.onclick = () => loadRun(r.run_id);
        list.appendChild(item);
    });
}

async function loadRun(runId) {
    document.getElementById('statusTxt').innerText = "Loading from disk...";
    const res = await fetch(`/api/runs/${runId}`);
    currentData = await res.json();
    document.getElementById('promptInput').value = currentData.prompt;
    document.getElementById('answerOut').innerText = "Model Output: " + currentData.answer;
    document.getElementById('statusTxt').innerText = `Loaded ${runId}`;
    renderMatrix();
}

async function runNewInference(overridePrompt = null) {
    const prompt = overridePrompt || document.getElementById('promptInput').value;
    const btn = document.getElementById('runBtn');
    const status = document.getElementById('statusTxt');
    
    btn.disabled = true;
    status.innerText = "Waking Modal GPU & Computing...";
    document.getElementById('matrixBox').innerHTML = '<div style="text-align:center; padding:50px;">Processing on Modal A100...</div>';

    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt: prompt, max_new_tokens: 12 })
        });
        currentData = await res.json();
        document.getElementById('answerOut').innerText = "Model Output: " + currentData.answer;
        status.innerText = "Complete. Saved locally.";
        await loadHistory();
        renderMatrix();
    } catch (e) {
        status.innerText = "Error: " + e.message;
    } finally {
        btn.disabled = false;
    }
}

function switchView(mode) {
    currentMode = mode;
    document.getElementById('btnTmpl').className = mode === 'template' ? 'active' : '';
    document.getElementById('btnTok').className = mode === 'token' ? 'active' : '';
    if (currentData) renderMatrix();
}

function renderMatrix() {
    if (!currentData) return;
    const grid = currentMode === 'template' ? currentData.template_grid : currentData.token_grid;
    const tokens = currentData.tokens;
    
    let html = '<table><thead><tr><th>L \\ T</th>';
    tokens.forEach((tok, i) => {
        const isGen = i >= currentData.prompt_length ? "gen-token" : "";
        html += `<th class="token-hdr ${isGen}" onclick="openSwapModal(${i}, '${escapeHtml(tok)}')">${escapeHtml(tok)}<br><span style="color:#8b949e; font-size:9px;">#${i}</span></th>`;
    });
    html += '</tr></thead><tbody>';

    for (let l = 0; l < currentData.num_layers; l++) {
        html += `<tr><td style="font-weight:bold; background:#161b22;">L${l}</td>`;
        for (let p = 0; p < tokens.length; p++) {
            const cell = grid[l][p];
            const alpha = Math.min(1, Math.max(0.12, cell.top_prob));
            const color = currentMode === 'template' ? `rgba(47, 129, 247, ${alpha})` : `rgba(137, 87, 229, ${alpha})`;
            
            html += `<td class="cell" style="background:${color}" onmouseover='inspectCell(${l}, ${p}, ${JSON.stringify(cell.top_k)})'>
                ${escapeHtml(cell.top_label)}
            </td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('matrixBox').innerHTML = html;
}

function inspectCell(l, p, topK) {
    let html = `<div style="margin-bottom:8px;"><strong>Layer:</strong> ${l} | <strong>Pos:</strong> ${p}</div>`;
    topK.forEach(item => {
        const pct = (item.prob * 100).toFixed(1);
        html += `<div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding:4px 0;">
            <span style="max-width:200px; word-break:break-word;"><code>${escapeHtml(item.label)}</code></span>
            <span style="color:#58a6ff;">${pct}%</span>
        </div>`;
    });
    document.getElementById('inspectorBody').innerHTML = html;
}

let activeSwapPos = null;
function openSwapModal(pos, tok) {
    if (pos >= currentData.prompt_length) {
        alert("Generated tokens cannot be swapped. Edit the prompt instead.");
        return;
    }
    activeSwapPos = pos;
    document.getElementById('swapPos').innerText = pos;
    document.getElementById('swapInput').value = tok.trim();
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('swapModal').style.display = 'block';
}

function closeSwapModal() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('swapModal').style.display = 'none';
}

function executeTokenSwap() {
    const newText = document.getElementById('swapInput').value;
    closeSwapModal();
    let newPrompt = "";
    for (let i = 0; i < currentData.prompt_length; i++) {
        if (i === activeSwapPos) newPrompt += newText;
        else newPrompt += currentData.tokens[i];
    }
    document.getElementById('promptInput').value = newPrompt;
    runNewInference(newPrompt);
}

function escapeHtml(s) { return s ? s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;") : ''; }

window.onload = loadHistory;
</script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
