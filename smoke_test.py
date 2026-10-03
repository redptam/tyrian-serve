import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Configuration
# Using absolute paths to avoid issues
current_dir = os.path.abspath(".")
model_path = os.path.join(current_dir, ".modelcache")

# Add the model cache to the python path so relative imports in modeling_tyrian.py work
if model_path not in sys.path:
    sys.path.append(model_path)

# Request context size: 8192 (User requested this, though training was 2048)
num_ctx = 8192

print(f"Loading model from {model_path}...")
# Loading model with trust_remote_code=True as required by Tyrian architecture
# We use bfloat16 as per config, but since this is CPU, we might want to check support.
# For now, let's stick to bfloat16.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    dtype=torch.bfloat16,
    device_map="cpu"
)

# Patching max_seq_len and recomputing RoPE buffers for 8192 context
# The model's precompute_rope_freqs uses config.max_seq_len
# We need to manually set the config and register the buffers
import modeling_tyrian
from modeling_tyrian import precompute_rope_freqs

head_dim = model.config.hidden_size // model.config.num_heads
# Recompute RoPE buffers for 8192 context
cos, sin = precompute_rope_freqs(head_dim, num_ctx, theta=model.config.rope_theta)
model.rope_cos = cos.to(torch.bfloat16)
model.rope_sin = sin.to(torch.bfloat16)

# Note: We also need to update the config so the model uses our new max_seq_len
# in its own forward pass if it uses self.config.max_seq_len.
model.config.max_seq_len = num_ctx

print("Model loaded and patched for 8192 context.")

# Test generation
prompt = "Hello! My name is Tyrian."
messages = [
    {"role": "user", "content": prompt}
]

# Use the chat template from the tokenizer
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

print(f"Input text: {input_text}")
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )

# Extract generated text
generated_ids = outputs[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Response: {response}")

# Verification
if response:
    print("Smoke test passed!")
else:
    print("Smoke test failed: No response generated.")
