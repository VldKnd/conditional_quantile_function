import torch


selected_params = {
    "bio": {
        "hidden_dimension": 12,
        "number_of_hidden_layers": 4,
        "batch_size": 512,
        "n_epochs": 100,
        "warmup_iterations": 10,
        "learning_rate": 0.01,
        "dtype": torch.float32,
    },
    "blog": {
        "hidden_dimension": 16,
        "number_of_hidden_layers": 4,
        "batch_size": 512,
        "n_epochs": 50,
        "warmup_iterations": 10,
        "learning_rate": 0.01,
        "dtype": torch.float32,
    },
    "sgemm": {
        "hidden_dimension": 46,
        "number_of_hidden_layers": 4,
        "batch_size": 8192,
        "n_epochs": 150,
        "warmup_iterations": 10,
        "learning_rate": 0.01,
        "dtype": torch.float32,
    },
    'rf1': {
        'learning_rate': 0.001,
        'batch_size': 512,
        'n_epochs': 500,
        'warmup_iterations': 50,
        'hidden_dimension': 8,
        'number_of_hidden_layers': 1,
        "dtype": torch.float32,
    },
    'rf2': {
        'learning_rate': 0.001,
        'batch_size': 2048,
        'n_epochs': 500,
        'warmup_iterations': 50,
        'hidden_dimension': 8,
        'number_of_hidden_layers': 2,
        "dtype": torch.float32,
    },
    'scm1d': {
        'learning_rate': 0.01,
        'batch_size': 2048,
        'n_epochs': 500,
        'warmup_iterations': 50,
        'hidden_dimension': 6,
        'number_of_hidden_layers': 3,
        "dtype": torch.float32,
    },
    'scm20d': {
        'learning_rate': 0.0001,
        'batch_size': 256,
        'n_epochs': 500,
        'warmup_iterations': 50,
        'hidden_dimension': 10,
        'number_of_hidden_layers': 1,
        "dtype": torch.float32,
    }
}
