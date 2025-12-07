import torch
from torch import nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss.

    Both `pred` and `target` have the same shape. Return a scalar tensor.
    """
    # TODO: implement MSE loss using basic tensor ops (no nn.MSELoss)
    return torch.mean((pred - target) ** 2)


class SimpleMLP(nn.Module):
    """
    Tiny MLP for practicing nn.Module implementation.

    Architecture (for example):
        input_dim -> hidden_dim -> ReLU -> output_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # TODO: define layers (e.g. self.fc1, self.fc2)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, output_dim)
        """
        # TODO: implement forward pass
        h1 = self.relu(self.fc1(x))
        out = self.fc2(h1)
        return out


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    One training step:
    - forward
    - compute loss (MSE)
    - backward
    - optimizer step
    - zero gradients

    Returns:
        loss_value as a Python float
    """
    # TODO: implement training step
    y_pred = model(x)
    loss = mse_loss(y_pred, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def tiny_training_loop() -> list[float]:
    """
    Practice:
    - creating a model
    - looping for a few iterations
    - printing loss values
    """
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    # Simple linear relation with noise
    true_w = torch.tensor([[2.0, -1.0, 0.5, 3.0]]).T  # (4, 1)
    true_b = torch.tensor([0.3])
    y = x @ true_w + true_b  # (32, 1)

    model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # TODO: run e.g. 50 training steps, calling training_step in a loop
    # Print loss every 10 steps
    NUMBER_EPOCHS = 50
    loss_values = []
    for epoch in range(NUMBER_EPOCHS):
        loss = training_step(model, optimizer, x, y)
        loss_values.append(loss)
        print(f"{epoch=}, {loss=}")

    return loss_values


if __name__ == "__main__":
    print("Running basic checks for 02_modules_and_training_loop.py")

    try:
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
        x = torch.randn(5, 4)
        y = model(x)
        assert y.shape == (5, 2)
        print("SimpleMLP forward shape check passed:", y.shape)
    except NotImplementedError:
        print("SimpleMLP: TODOs not implemented yet.")

    try:
        # If tiny_training_loop returns a list/tuple of losses, check that they decrease.
        result = tiny_training_loop()
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            start_loss = float(result[0])
            end_loss = float(result[-1])
            assert end_loss <= start_loss, "Loss did not decrease between first and last step"
            print("tiny_training_loop loss decrease check passed:", start_loss, "->", end_loss)
        else:
            print(
                "tiny_training_loop ran without errors. "
                "Optionally return a list of loss values to enable automatic loss checks."
            )
    except NotImplementedError:
        print("tiny_training_loop: TODO not implemented yet.")
