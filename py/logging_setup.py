import os
import yaml
import logging.config
from pathlib import Path

def setup_logging(
    config_path: str = "config/logging_config.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG"
) -> None:
    """Setup logging configuration"""
    path = os.getenv(env_key, config_path)
    if os.path.exists(path):
        with open(path, "rt") as f:
            try:
                config = yaml.safe_load(f.read())
                # Ensure log directories exist
                for handler in config.get("handlers", {}).values():
                    if "filename" in handler:
                        log_dir = os.path.dirname(handler["filename"])
                        Path(log_dir).mkdir(parents=True, exist_ok=True)
                # Apply configuration
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging configuration: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Could not find logging config file at {path}")

# Example usage in different modules:

# In verification module:
logger = logging.getLogger("decentralized_gpu.verification")

def verify_proof():
    logger.debug("Starting proof verification")
    try:
        # Verification logic
        logger.info("Proof verified successfully")
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)

# In payment module:
logger = logging.getLogger("decentralized_gpu.payment")

def process_payment():
    logger.debug("Initiating payment processing")
    try:
        # Payment logic
        logger.info("Payment processed successfully")
    except Exception as e:
        logger.error(f"Payment failed: {str(e)}", exc_info=True)

# In performance monitoring:
logger = logging.getLogger("decentralized_gpu.performance")

def log_performance_metrics(metrics: dict):
    logger.info("Performance metrics", extra=metrics)

# Example application startup:
if __name__ == "__main__":
    setup_logging()
    
    # Now all modules will use the configured logging
    logger = logging.getLogger("decentralized_gpu")
    logger.info("Application starting")