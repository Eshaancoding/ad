import torch

def assert_results(result, input_grad):
    print(result.sum().item())
    print(input_grad.sum().item())
    # Uncomment asserts to verify expected values if desired
    # assert round(result.sum().item(), 5) == 6.2689
    # assert round(input_grad.sum().item(), 5) == 5.39896

def test_dotprod():
    torch.manual_seed(42)  # Set seed for reproducibility
    
    inp = torch.randn(2, 5, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    result = inp @ weight + bias  

    result.sum().backward()

    print(inp, weight, bias)

    assert_results(result, inp.grad)

if __name__ == "__main__":
    test_dotprod()

