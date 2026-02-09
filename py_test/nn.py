
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Custom Sequential defined to enforce manual initialization
class CustomSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

        # Initialize every parameter with 0.5
        for module in self:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)

def benchmark(fn, name="Benchmark"):
    start = time.time()
    fn()
    end = time.time()
    print(f"{name}: {end - start:.6f} seconds")

def test_nn():
    # Build NN model: Linear(256->256), Sigmoid, Linear(256->256)
    nn_model = CustomSequential(
        nn.Linear(256, 256),
        nn.Sigmoid(),
        nn.Linear(256, 256)
    )

    # Define optimizer (SGD with lr=0.01)
    opt = optim.SGD(nn_model.parameters(), lr=0.01)

    # Training step function
    def f():
        opt.zero_grad()

        # Input tensor filled with 0.2
        val = torch.full((2, 256), 0.2, dtype=torch.float32)

        # Forward pass
        res = nn_model(val)

        # Dummy loss = sum of outputs
        loss = res.sum()

        # Backward pass
        loss.backward()

        # Update step
        opt.step()

    # Benchmark param-tracking loop
    def loop_fn():
        for _ in range(1000):
            f()

    benchmark(loop_fn, name="Tracking nodes")

    # Inspect parameters
    print("Saving params:")
    with torch.no_grad():
        for name, param in nn_model.named_parameters():
            print(name, param.flatten()[:5], "...")  # print first 5 values

    # Benchmark single execution
    benchmark(f, name="full exec")

if __name__ == "__main__":
    test_nn()

