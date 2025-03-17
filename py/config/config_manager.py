import os
import yaml
import logging
import logging.config
from typing import Dict, Any

class ConfigManager:
    """Manages configuration loading, merging, and access for the system."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration manager."""
        self.config_dir = config_dir
        self._config = {}
        self.logger = None
    
    def load_configuration(self) -> Dict[str, Any]:
        """
        Load and merge all configuration files.
        
        Returns:
            Dict containing the merged configuration.
        """
        # Initialize empty configuration
        config = {}
        
        # Define configuration files to load
        config_files = [
            "default_config.yml",      # Base configuration
            "distribution_config.yml", # Data distribution configuration
            "logging_config.yml"       # Logging configuration
        ]
        
        # Try to load local_config.yml if it exists (overrides defaults)
        local_config_path = os.path.join(self.config_dir, "local_config.yml")
        if os.path.exists(local_config_path):
            config_files.append("local_config.yml")
        
        # Load each configuration file
        for config_file in config_files:
            file_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                        
                    # Handle special integration based on file type
                    if config_file == "distribution_config.yml":
                        config['distribution'] = file_config
                    elif config_file == "logging_config.yml":
                        config['logging_config'] = file_config
                    elif config_file == "local_config.yml":
                        # Deep merge local config
                        config = self._deep_merge(config, file_config)
                    else:
                        # Regular merge for default config
                        config.update(file_config)
                        
                except Exception as e:
                    print(f"Error loading {config_file}: {str(e)}")
            else:
                print(f"Warning: Configuration file {config_file} not found")
        
        # Store the merged config
        self._config = config
        
        # Set up logging after configuration is loaded
        self._setup_logging()
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        if not self._config:
            return self.load_configuration()
        return self._config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two dictionaries, with override values taking precedence.
        
        Args:
            base: Base dictionary
            override: Dictionary with override values
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override or add value
                result[key] = value
                
        return result
    
    def _setup_logging(self):
        """Configure the logging system using the loaded configuration."""
        try:
            # Use the logging config if available
            if 'logging_config' in self._config:
                logging.config.dictConfig(self._config['logging_config'])
                self.logger = logging.getLogger('decentralized_gpu')
                self.logger.info("Logging configured from logging_config.yml")
            # Otherwise use basic config from main configuration
            elif 'logging' in self._config:
                log_config = self._config['logging']
                logging.basicConfig(
                    level=getattr(logging, log_config.get('level', 'INFO')),
                    format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                    filename=log_config.get('file'),
                    filemode='a'
                )
                self.logger = logging.getLogger('decentralized_gpu')
                self.logger.info("Logging configured from main config")
            else:
                # Fallback to basic config
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                self.logger = logging.getLogger('decentralized_gpu')
                self.logger.warning("Using default logging configuration")
                
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            # Ensure basic logging is available
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('decentralized_gpu')
    
    def reload_configuration(self) -> Dict[str, Any]:
        """Reload all configuration files."""
        self._config = {}
        return self.load_configuration()

    def get_distribution_config(self) -> Dict[str, Any]:
        """Get the data distribution configuration section."""
        return self._config.get('distribution', {})
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get the network configuration section."""
        return self._config.get('network', {})
    
    def get_wallet_config(self) -> Dict[str, Any]:
        """Get the wallet configuration section."""
        return self._config.get('wallet', {})
    
    def get_verification_config(self) -> Dict[str, Any]:
        """Get the verification configuration section."""
        return self._config.get('verification', {})

    def get_sybil_protection_config(self) -> Dict[str, Any]:
        """Get the Sybil protection configuration section."""
        return self._config.get('sybil_protection', {})


# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.load_configuration()
    print("Configuration loaded successfully")
    
    # Access specific configuration sections
    network_config = config_manager.get_network_config()
    distribution_config = config_manager.get_distribution_config()
    
    logger = logging.getLogger('decentralized_gpu')
    logger.info("Configuration test complete")