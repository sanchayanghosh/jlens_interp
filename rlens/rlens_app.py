import modal
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pydantic

# ---------------------------------------------------------------------------
# Modal Infrastructure Setup
# ---------------------------------------------------------------------------
app = modal.App("qwen-rlens-template-explorer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.0.0",
        "hf_transfer",
        "huggingface_hub",
        "safetensors"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# Configure your Hugging Face Lens repos here once you have access
LENS_REPO_ID = "camilablank/workspace-lenses" 
RLENS_FILENAME = "qwen-32b-rlens.safetensors"
TEMPLATE_WEIGHTS_FILENAME = "template_lens_weights.safetensors"
TEMPLATE_LABELS_FILENAME = "template_lens_phrases.json"

# ---------------------------------------------------------------------------
# Serverless GPU Engine (Shuts off 10s after request finishes)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A100-80GB", 
    image=image,
    timeout=600,
    scaledown_window=10, 
)
class TemplateRLensEngine:
    @modal.enter()
    def load_model(self):
        import torch
        import json
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print(f"Spinning up GPU and loading {MODEL_ID} in bfloat16...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # 1. Try Loading R-Lens
        self.r_lenses = None
        try:
            r_lens_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=RLENS_FILENAME)
            raw_r_weights = load_file(r_lens_path)
            self.r_lenses = {}
            for l in range(len(self.model.model.layers)):
                self.r_lenses[l] = {
                    "weight": raw_r_weights[f"layer_{l}.weight"].to(self.model.device, dtype=torch.bfloat16),
                    "bias": raw_r_weights.get(f"layer_{l}.bias", torch.zeros(self.model.config.hidden_size)).to(self.model.device, dtype=torch.bfloat16)
                }
            print("Loaded R-Lens matrices.")
        except Exception as e:
            print(f"Could not load R-Lens ({e}). Falling back to Identity (J-Lens).")

        # 2. Try Loading Template Lens (Phrase Dictionary)
        self.template_lens = None
        try:
            t_weights_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=TEMPLATE_WEIGHTS_FILENAME)
            t_labels_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=TEMPLATE_LABELS_FILENAME)
            
            self.template_lens = {
                "weight": load_file(t_weights_path)["weight"].to(self.model.device, dtype=torch.bfloat16)
            }
            with open(t_labels_path, "r") as f:
                self.template_lens["labels"] = json.load(f) # List of string phrases
            print("Loaded Template Lens phrases.")
        except Exception as e:
            print(f"Could not load Template Lens ({e}). Falling back to native W_U (Tokens).")

    @modal.method()
    def analyze_trajectory(self, prompt: str, max_new_tokens: int = 15, top_k: int = 5):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Forward pass returning hidden states for all generated tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_token_ids = outputs.sequences[0]
        token_strings = [self.tokenizer.decode([t]) for t in full_token_ids]
        num_layers = len(self.model.model.layers)
        grid_data = []

        # Extract hidden states across all layers
        for l in range(num_layers):
            layer_row = []
            prompt_hs = outputs.hidden_states[0][l+1]
            gen_hs = [outputs.hidden_states[i][l+1] for i in range(1, len(outputs.hidden_states))] if len(outputs.hidden_states) > 1 else []
            full_hs = torch.cat([prompt_hs, torch.cat(gen_hs, dim=1)], dim=1) if gen_hs else prompt_hs
            
            x_l = full_hs[0] # Shape: (seq_len, hidden_dim)

            # --- 1. Apply R-Lens (Propagation) ---
            if self.r_lenses:
                r_w = self.r_lenses[l]["weight"]
                r_b = self.r_lenses[l]["bias"]
                x_l = torch.matmul(x_l, r_w.T) + r_b

            normed = self.model.model.norm(x_l)

            # --- 2. Apply Template Lens (Unembedding to Phrases) ---
            if self.template_lens:
                logits = torch.matmul(normed, self.template_lens["weight"].T)
                labels = self.template_lens["labels"]
            else:
                # Fallback to standard tokens
                logits = self.model.lm_head(normed)
                labels = None

            probs = torch.softmax(logits.float(), dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

            for pos in range(len(token_strings)):
                top_items = []
                for p, idx in zip(top_probs[pos], top_indices[pos]):
                    # If we have template labels, use the phrase. Otherwise, use tokenizer.
                    text = labels[idx.item()] if labels else self.tokenizer.decode([idx.item()])
                    top_items.append({
                        "phrase": text,
                        "prob": round(p.item(), 4),
                    })
                
                layer_row.append({
                    "top_phrase": top_items[0]["phrase"],
                    "top_prob": top_items[0]["prob"],
                    "top_k": top_items,
                })
            grid_data.append(layer_row)

        final_answer = self.tokenizer.decode(outputs.sequences[0][inputs["input_ids"].shape[1]:])

        return {
            "tokens": token_strings,
            "prompt_length": inputs["input_ids"].shape[1],
            "num_layers": num_layers,
            "grid": grid_data,
            "answer": final_answer
        }


# ---------------------------------------------------------------------------
# Lightweight Web Server (Runs on CPU, stores state in browser)
# ---------------------------------------------------------------------------
web_app = FastAPI()

class PromptRequest(pydantic.BaseModel):
    prompt: str

@web_app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Template + R-Lens Inspector</title>
    <style>
        :root {
            --bg-main: #0e1117; --bg-card: #161b22; --border: #30363d;
            --text-main: #e6edf3; --text-muted: #8b949e; --accent: #8957e5;
        }
        * { box-sizing: border-box; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
        body { background-color: var(--bg-main); color: var(--text-main); margin: 0; padding: 24px; }
        .container { max-width: 1600px; margin: auto; }
        .header { display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); padding: 16px; border-radius: 6px; margin-bottom: 20px; }
        textarea { width: 100%; height: 60px; background: #010409; color: white; border: 1px solid var(--border); padding: 10px; border-radius: 4px; }
        button { background: var(--accent); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; margin-top: 10px; }
        button:hover { background: #9e6cf2; }
        .grid-wrapper { display: flex; gap: 20px; }
        .matrix { overflow-x: auto; flex: 1; border: 1px solid var(--border); background: var(--bg-card); padding: 10px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid var(--border); padding: 6px; text-align: center; font-size: 11px; white-space: nowrap; }
        th { background: #1f242c; position: sticky; top: 0; }
        th.token-header { cursor: pointer; transition: background 0.2s; }
        th.token-header:hover { background: #30363d; outline: 1px solid #58a6ff; }
        td.cell { cursor: crosshair; max-width: 120px; overflow: hidden; text-overflow: ellipsis; }
        .sidebar { width: 340px; background: var(--bg-card); border: 1px solid var(--border); padding: 16px; border-radius: 6px; }
        .gen-token { border-bottom: 2px solid #58a6ff; }
        .modal { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--bg-card); border: 1px solid var(--accent); padding: 20px; border-radius: 8px; z-index: 100; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 99; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>🟣 Template Lens + R-Lens Explorer (Qwen-32B)</h2>
        <span style="color:var(--text-muted); font-size: 12px; margin-top:10px;">Status: GPU Shuts off automatically when idle. State saved locally.</span>
    </div>

    <div class="card">
        <label>Input Prompt:</label>
        <textarea id="promptInput">The Eiffel Tower is located in the city of</textarea>
        <button id="runBtn" onclick="runLens()">Calculate Semantic Trajectory</button>
        <div id="answerOut" style="margin-top: 10px; color: #58a6ff; font-weight: bold;"></div>
    </div>

    <div class="grid-wrapper">
        <div class="matrix" id="matrix">
            <div style="text-align:center; padding:50px; color:var(--text-muted);">Awaiting Run...</div>
        </div>
        <div class="sidebar">
            <h3 style="margin-top:0;">Phrase Readout</h3>
            <div id="tooltip">Hover over any cell in the grid to view the top multi-token phrase projections.</div>
        </div>
    </div>
</div>

<!-- Intervention Modal -->
<div class="overlay" id="overlay" onclick="closeModal()"></div>
<div class="modal" id="swapModal">
    <h3>Steer / Swap Prompt Token</h3>
    <p style="font-size: 12px; color: var(--text-muted);">Edit this segment to trigger a counterfactual run.</p>
    <div style="margin-bottom: 10px;">
        <label>Replacing at Pos <span id="swapPos"></span>:</label>
        <input type="text" id="swapInput" style="width: 100%; background: #010409; color: white; border: 1px solid var(--border); padding: 8px; margin-top: 5px;" />
    </div>
    <button onclick="executeSwap()" style="background: var(--accent);">Execute Counterfactual Run</button>
</div>

<script>
let currentTokens = [];
let promptLength = 0;

async function runLens(overridePrompt = null) {
    const btn = document.getElementById('runBtn');
    const prompt = overridePrompt || document.getElementById('promptInput').value;
    
    btn.disabled = true;
    btn.innerText = "Waking GPU & Computing... (May take 60s on cold start)";
    document.getElementById('matrix').innerHTML = `<div style="text-align:center; padding:50px;">Running forward pass & generating readouts...</div>`;

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        
        // Save the full readout locally in the UI
        currentTokens = data.tokens;
        promptLength = data.prompt_length;
        
        document.getElementById('answerOut').innerText = "Model Final Output: " + data.answer;
        renderGrid(data);
    } catch (e) {
        document.getElementById('matrix').innerHTML = `<div style="color:#f85149; padding:20px;">Error: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Calculate Semantic Trajectory";
    }
}

function renderGrid(data) {
    let html = '<table><thead><tr><th>L \\ T</th>';
    
    data.tokens.forEach((tok, i) => {
        const isGen = i >= data.prompt_length ? "gen-token" : "";
        const title = isGen ? "Generated Token" : "Click to Swap Prompt Token";
        html += `<th class="token-header ${isGen}" title="${title}" onclick="openSwapModal(${i}, '${escapeHtml(tok)}')">
                    ${escapeHtml(tok)}<br><span style="color:#8b949e; font-size:9px;">#${i}</span>
                 </th>`;
    });
    html += '</tr></thead><tbody>';

    for (let l = 0; l < data.num_layers; l++) {
        html += `<tr><td style="font-weight:bold; background:#161b22;">L${l}</td>`;
        for (let p = 0; p < data.tokens.length; p++) {
            const cell = data.grid[l][p];
            const alpha = Math.min(1, Math.max(0.15, cell.top_prob));
            const bg = `rgba(137, 87, 229, ${alpha})`;
            
            html += `<td class="cell" style="background:${bg}" onmouseover='showTooltip(${JSON.stringify(cell.top_k)})'>
                ${escapeHtml(cell.top_phrase)}
            </td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('matrix').innerHTML = html;
}

function showTooltip(topK) {
    let html = '';
    topK.forEach(item => {
        const pct = (item.prob * 100).toFixed(1);
        html += `<div style="display:flex; justify-content:space-between; border-bottom:1px solid #30363d; padding:6px 0;">
                    <span style="max-width: 250px; white-space: normal; word-wrap: break-word;"><code>${escapeHtml(item.phrase)}</code></span>
                    <span style="color:var(--accent); margin-left: 10px;">${pct}%</span>
                 </div>`;
    });
    document.getElementById('tooltip').innerHTML = html;
}

let swapTargetPos = null;

function openSwapModal(pos, currentText) {
    if (pos >= promptLength) {
        alert("Cannot swap a generated token directly. You must edit the input prompt.");
        return;
    }
    swapTargetPos = pos;
    document.getElementById('swapPos').innerText = pos;
    document.getElementById('swapInput').value = currentText.trim();
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('swapModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('swapModal').style.display = 'none';
}

function executeSwap() {
    const newText = document.getElementById('swapInput').value;
    closeModal();
    
    // Reconstruct the prompt
    let newPrompt = "";
    for(let i = 0; i < promptLength; i++) {
        if(i === swapTargetPos) newPrompt += newText;
        else newPrompt += currentTokens[i];
    }
    
    document.getElementById('promptInput').value = newPrompt;
    runLens(newPrompt); // Sends req to Modal, waking GPU
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
</script>
</body>
</html>
    """

@web_app.post("/api/analyze")
def analyze_endpoint(req: PromptRequest):
    engine = TemplateRLensEngine()
    result = engine.analyze_trajectory.remote(req.prompt)
    return JSONResponse(content=result)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
