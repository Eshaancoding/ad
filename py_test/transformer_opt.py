import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Transformer Encoder definition ---
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,   # aligns with your (batch, seq, feature)
            activation="relu",  # closer to modern transformer practice
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x expected: (batch, seq, d_model)
        return self.encoder(x)

# --- Model, optimizer, compile ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleTransformerEncoder(
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_layers=1
).to(device)

opt = optim.SGD(model.parameters(), lr=0.01)

# Wrap forward/backward+step in a single function
def step_fn():
    x = torch.full((2, 64), 0.3, dtype=torch.float32, device=device)  
    # shaped as (batch=2, seq=64, d_model=64)
    out = model(x)

    loss = out.sum() / 1000.0   # simple dummy objective
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss

# Compile the training step for speed (PyTorch 2.0+)
compiled_step = torch.compile(step_fn, mode="max-autotune")

# --- Benchmarking ---
import time
num_steps = 3000

start = time.time()
for i in range(num_steps):
    loss = compiled_step()
    if (i+1) % 100 == 0:
        print(f"Step {i+1}: loss={loss.item():.4f}")
end = time.time()

print(f"Full execution time for {num_steps} steps: {end - start:.2f}s")

# --- Final Parameter Saving ---
with torch.no_grad():
    total_params = 0.0
    num_tensors = 0
    for p in model.parameters():
        total_params += p.sum().item()
        num_tensors += 1
    print("Number of tensors:", num_tensors)
    print("Sum of all parameters:", total_params)
