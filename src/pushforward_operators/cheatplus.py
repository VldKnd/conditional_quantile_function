import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Module


class CheatPlusFunction(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input: torch.Tensor, beta=1.0, threshold=20.0) -> torch.Tensor:
        output = F.softplus(input, beta=beta, threshold=threshold) + F.softplus(-input, beta=beta, threshold=threshold)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        input, beta, threshold = inputs
        ctx.save_for_backward(input)
        ctx.beta = beta
        ctx.threshold = threshold

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, = ctx.saved_tensors
        grad_input = None

        beta = ctx.beta
        threshold = ctx.threshold

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = input.clone()
            mask = (input * beta > threshold) | (-input * beta < -threshold)
            grad_input[mask] = grad_output[mask]
            grad_input[~mask] = grad_output[~mask] * torch.tanh(input[~mask] / 2)

        return grad_input, None, None


class CheatPlus(Module):
    __constants__ = ["beta", "threshold"]
    beta: float
    threshold: float

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CheatPlusFunction.apply(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


def cheatplus(x):
    result = CheatPlusFunction.apply(x)
    return result
