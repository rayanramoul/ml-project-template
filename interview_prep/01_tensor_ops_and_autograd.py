import torch
import torch.nn.functional as F


def manual_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simple linear layer: y = x @ w.T + b

    Args:
        x: input tensor of shape (batch, in_features)
        w: weight tensor of shape (out_features, in_features)
        b: bias tensor of shape (out_features,)

    Returns:
        Tensor of shape (batch, out_features)
    """
    # Implement linear layer using matrix multiplication and broadcasting.
    # x: (batch, in_features)
    # w: (out_features, in_features) -> w.T: (in_features, out_features)
    # b: (out_features,) broadcasts over the batch dimension.
    return x @ w.T + b


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss.

    Both `pred` and `target` have the same shape. Return a scalar tensor.
    """
    # Compute elementwise squared error and average over all elements.
    return torch.mean((pred - target) ** 2)


def gradient_step_example() -> list[float]:
    """
    Small example to practice:
    - creating parameters with requires_grad=True
    - doing a forward pass
    - calling backward()
    - performing a gradient descent step
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    target = torch.randn(4, 2)

    # Initialize parameters as leaf tensors so autograd tracks their gradients.
    w = torch.randn(2, 3, requires_grad=True)  # (out_features=2, in_features=3)
    b = torch.zeros(2, requires_grad=True)     # (out_features,)

    # TODO:
    # 1. Call manual_linear
    # 2. Compute loss with mse_loss
    # 3. Call backward()
    # 4. Update w and b with a small learning rate (e.g. 0.1) *without* using an optimizer
    # 5. Zero the gradients manually
    losses: list[float] = []
    lr = 0.1

    for _ in range(2):
        # Forward pass
        y_pred = manual_linear(x, w, b)
        # Compute scalar loss
        loss = mse_loss(y_pred, target)
        losses.append(float(loss))

        # Backward pass: compute gradients dloss/dw and dloss/db
        loss.backward()

        # Gradient descent update under torch.no_grad so we don't track these ops
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            # Zero gradients to avoid accumulation across steps
            w.grad.zero_()
            b.grad.zero_()

    return losses


def reshape_vs_view_example(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Practice thinking about `view` vs `reshape`.

    Args:
        x: tensor of shape (batch, seq_len, d_model)

    Returns:
        A tuple (y_view, y_reshape) where:
        - y_view is a tensor of shape (batch * seq_len, d_model) created using .view(...)
        - y_reshape is the same logical shape using .reshape(...)

    During the interview, be ready to discuss:
    - When .view() can fail (non-contiguous tensors)
    - Why .reshape() is more flexible (may copy data)
    """
    # TODO: implement using .view and .reshape
    batch, seq_len, d_model = x.shape
    y_view = x.contiguous().view((batch * seq_len, d_model))
    y_reshape = x.reshape((batch * seq_len, d_model))
    return (y_view, y_reshape)


if __name__ == "__main__":
    print("Running basic checks for 01_tensor_ops_and_autograd.py")

    x = torch.randn(2, 3)
    w = torch.randn(4, 3)
    b = torch.randn(4)

    try:
        y = manual_linear(x, w, b)
        assert y.shape == (2, 4)
        # Compare against the reference PyTorch expression
        y_ref = x @ w.T + b
        assert torch.allclose(y, y_ref, atol=1e-6), "manual_linear does not match x @ w.T + b"
        print("manual_linear shape + value check passed:", y.shape)
    except NotImplementedError:
        print("manual_linear: TODO not implemented yet.")

    pred = torch.randn(5, 2)
    target = torch.randn(5, 2)
    try:
        loss = mse_loss(pred, target)
        assert loss.shape == ()
        # Compare against torch.nn.functional.mse_loss
        loss_ref = F.mse_loss(pred, target)
        assert torch.allclose(loss, loss_ref, atol=1e-6), "mse_loss does not match F.mse_loss"
        print("mse_loss scalar + value check passed:", loss.item())
    except NotImplementedError:
        print("mse_loss: TODO not implemented yet.")

    x = torch.randn(2, 3, 4)
    try:
        y_view, y_reshape = reshape_vs_view_example(x)
        assert y_view.shape == (2 * 3, 4)
        assert y_reshape.shape == (2 * 3, 4)
        # Check that both are consistent with a simple flatten over (batch, seq_len)
        flat_ref = x.reshape(-1, x.size(-1))
        assert torch.allclose(y_view, flat_ref), "y_view does not match flatten reference"
        assert torch.allclose(y_reshape, flat_ref), "y_reshape does not match flatten reference"
        print("reshape_vs_view_example shape + value check passed:", y_view.shape)
    except NotImplementedError:
        print("reshape_vs_view_example: TODO not implemented yet.")

    # Optional: if you make gradient_step_example return a list/tuple of losses,
    # we can assert that the loss decreases.
    try:
        result = gradient_step_example()
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            start_loss = float(result[0])
            end_loss = float(result[-1])
            assert end_loss <= start_loss, "Loss did not decrease between steps in gradient_step_example"
            print("gradient_step_example loss decrease check passed:", start_loss, "->", end_loss)
        else:
            print(
                "gradient_step_example ran without errors. "
                "Optionally return [loss_start, loss_end, ...] to enable automatic loss checks."
            )
    except NotImplementedError:
        print("gradient_step_example: TODO not implemented yet.")
