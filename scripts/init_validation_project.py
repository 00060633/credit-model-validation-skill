import os
import yaml

# Define the directory structure
directories = [
    "data/raw",
    "data/processed",
    "data/model",
    "results/metrics",
    "results/plots",
    "results/reports"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Define configuration settings
config = {
    "model": {
        "name": "YourModelName",
        "version": "1.0",
        "parameters": {
            "learning_rate": 0.01,
            "num_epochs": 100,
            "batch_size": 32
        }
    },
    "data": {
        "input_path": "data/raw/",
        "output_path": "data/processed/"
    },
    "results": {
        "metrics_path": "results/metrics/",
        "plots_path": "results/plots/",
        "reports_path": "results/reports/"
    }
}

# Write the config to a YAML file
with open('config.yaml', 'w') as config_file:
    yaml.dump(config, config_file)