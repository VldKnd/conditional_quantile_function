import torch
import random
import numpy as np


def save_rng_state():
    """Save Python, NumPy, and PyTorch RNG states."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }


def restore_rng_state(state):
    """Restore Python, NumPy, and PyTorch RNG states."""
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


class RNGState:
    """Context manager to save and restore RNG states."""
    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        self.state = save_rng_state()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        restore_rng_state(self.state)


if __name__ == "__main__":
    print("Before:", torch.rand(1))

    # Functional usage
    state = save_rng_state()
    torch.manual_seed(42)
    print("With seed:", torch.rand(1))
    restore_rng_state(state)
    print("Restored:", torch.rand(1))

    # Context manager usage
    print("\nBefore (context):", torch.rand(1))
    with RNGState():
        torch.manual_seed(123)
        print("Inside context:", torch.rand(1))
    print("After context:", torch.rand(1))
