import modal

app = modal.App("qwen-lens-backend")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "safetensors",
        "huggingface_hub",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_ID = "Qwen/Qwen3.6-27B"
LENS_REPO_ID = "camilablank/workspace-lenses"
RLENS_FILENAME = "r_lens_qwen_3.6_27b.safetensors"
TEMPLATE_WEIGHTS_FILENAME = "template_lens_qwen_3.6_27b.safetensors"
TEMPLATE_LABELS_FILENAME = "template_lens_phrases.json"

@app.cls(
    gpu="A100-80GB",
    image=image,
    timeout=1200,
    scaledown_window=10,
    secrets=[modal.Secret.from_name("huggingface-secret")] # Your HF Token
)
class RLensEngine:


    # ---> ADD IT HERE <---
    @modal.method()
    def run_multiturn_trajectory(self, user_turns: list[str], max_new_tokens: int = 50, top_k: int = 5):
        import torch

        messages = []
        
        # 1. Iteratively generate the conversation
        for user_text in user_turns:
            messages.append({"role": "user", "content": user_text})
            
            # STEP A: Get the raw string formatted with the chat template (bypasses the bug)
            prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            # STEP B: Tokenize normally to get a clean, stable dictionary
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_length:]
            assistant_reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": assistant_reply})

        # 2. Get the final full conversation string and tokenize it
        full_prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        full_inputs = self.tokenizer(full_prompt_str, return_tensors="pt").to(self.model.device)
        
        # Decode using the inner input_ids array
        token_strings = [self.tokenizer.decode([t]) for t in full_inputs["input_ids"][0]]
        
        # 3. Do ONE forward pass to extract hidden states for the entire conversation
        with torch.no_grad():
            outputs = self.model(**full_inputs, output_hidden_states=True)
            
        num_layers = len(self.model.model.layers)
        token_grid = []

        # 4. Apply Lens Projections to all tokens across all layers
        for l in range(num_layers):
            layer_row = []
            x_l = outputs.hidden_states[l + 1][0] # Shape: (seq_len, hidden_dim)

            # Apply R-Lens Linear Transform if weights loaded
            if getattr(self, "r_lenses", None) is not None:
                r_w, r_b = self.r_lenses[l]["weight"], self.r_lenses[l]["bias"]
                x_l = torch.matmul(x_l, r_w.T) + r_b

            normed = self.model.model.norm(x_l)

            # Standard Projection
            logits = self.model.lm_head(normed)
            probs = torch.softmax(logits.float(), dim=-1)
            tp, ti = torch.topk(probs, k=top_k, dim=-1)

            for pos in range(len(token_strings)):
                top_k_items = [
                    {"label": self.tokenizer.decode([ti[pos][k].item()]), "prob": round(tp[pos][k].item(), 4)}
                    for k in range(top_k)
                ]
                layer_row.append({"top_label": top_k_items[0]["label"], "top_prob": top_k_items[0]["prob"], "top_k": top_k_items})

            token_grid.append(layer_row)

        return {
            "tokens": token_strings,
            "num_layers": num_layers,
            "answer": full_prompt_str, # Send the full text back to the UI
            "token_grid": token_grid,
            "template_grid": token_grid 
        }

    @modal.enter()
    def load_model(self):
        import json
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print(f"Loading {MODEL_ID} in bfloat16...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # 1. Load Precomputed R-Lens Projection Matrices
        self.r_lenses = None
        try:
            r_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=RLENS_FILENAME)
            raw_r = load_file(r_path)
            self.r_lenses = {}
            for l in range(len(self.model.model.layers)):
                self.r_lenses[l] = {
                    "weight": raw_r[f"layer_{l}.weight"].to(self.model.device, dtype=torch.bfloat16),
                    "bias": raw_r.get(f"layer_{l}.bias", torch.zeros(self.model.config.hidden_size)).to(self.model.device, dtype=torch.bfloat16)
                }
            print("Loaded R-Lens weights.")
        except Exception as e:
            print(f"R-Lens load error ({e}). Falling back to identity projection.")

        # 2. Load Precomputed Template Lens Phrase Dictionary
        self.template_lens = None
        try:
            tw_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=TEMPLATE_WEIGHTS_FILENAME)
            tl_path = hf_hub_download(repo_id=LENS_REPO_ID, filename=TEMPLATE_LABELS_FILENAME)
            self.template_lens = {
                "weight": load_file(tw_path)["weight"].to(self.model.device, dtype=torch.bfloat16)
            }
            with open(tl_path, "r") as f:
                self.template_lens["labels"] = json.load(f)
            print("Loaded Template Lens phrase weights.")
        except Exception as e:
            print(f"Template Lens load error ({e}). Template phrase mode disabled.")



    @modal.method()
    def run_lens_trajectory(self, prompt: str, max_new_tokens: int = 15, top_k: int = 5):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

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

        token_grid = []
        template_grid = []

        for l in range(num_layers):
            prompt_hs = outputs.hidden_states[0][l + 1]
            gen_hs = [outputs.hidden_states[i][l + 1] for i in range(1, len(outputs.hidden_states))] if len(outputs.hidden_states) > 1 else []
            full_hs = torch.cat([prompt_hs, torch.cat(gen_hs, dim=1)], dim=1) if gen_hs else prompt_hs
            
            x_l = full_hs[0]  # Shape: (seq_len, hidden_dim)

            # Apply R-Lens linear transform
            if self.r_lenses is not None:
                r_w = self.r_lenses[l]["weight"]
                r_b = self.r_lenses[l]["bias"]
                x_l = torch.matmul(x_l, r_w.T) + r_b

            normed = self.model.model.norm(x_l)

            # 1. Native Token Projection (R-Lens Token Lens)
            logits_tok = self.model.lm_head(normed)
            probs_tok = torch.softmax(logits_tok.float(), dim=-1)
            tp_tok, ti_tok = torch.topk(probs_tok, k=top_k, dim=-1)

            # 2. Template Phrase Projection (Template Lens)
            if self.template_lens is not None:
                logits_tmpl = torch.matmul(normed, self.template_lens["weight"].T)
                probs_tmpl = torch.softmax(logits_tmpl.float(), dim=-1)
                tp_tmpl, ti_tmpl = torch.topk(probs_tmpl, k=top_k, dim=-1)
            else:
                tp_tmpl, ti_tmpl = tp_tok, ti_tok

            tok_row = []
            tmpl_row = []
            for pos in range(len(token_strings)):
                # Parse token top-k
                top_k_toks = [
                    {"label": self.tokenizer.decode([ti_tok[pos][k].item()]), "prob": round(tp_tok[pos][k].item(), 4)}
                    for k in range(top_k)
                ]
                tok_row.append({"top_label": top_k_toks[0]["label"], "top_prob": top_k_toks[0]["prob"], "top_k": top_k_toks})

                # Parse phrase top-k
                if self.template_lens is not None:
                    top_k_phrases = [
                        {"label": self.template_lens["labels"][ti_tmpl[pos][k].item()], "prob": round(tp_tmpl[pos][k].item(), 4)}
                        for k in range(top_k)
                    ]
                else:
                    top_k_phrases = top_k_toks

                tmpl_row.append({"top_label": top_k_phrases[0]["label"], "top_prob": top_k_phrases[0]["prob"], "top_k": top_k_phrases})

            token_grid.append(tok_row)
            template_grid.append(tmpl_row)

        generated_text = self.tokenizer.decode(full_token_ids[prompt_len:])

        return {
            "tokens": token_strings,
            "prompt_length": prompt_len,
            "num_layers": num_layers,
            "answer": generated_text,
            "token_grid": token_grid,
            "template_grid": template_grid,
        }
