from train import train_full_pipeline
from config import get_config

if __name__ == "__main__":
    config = get_config()
    train_full_pipeline(config)