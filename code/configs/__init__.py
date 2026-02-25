"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import torch


class Config:
    """Configuration class that loads and manages all settings."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set derived attributes for easy access
        self._set_attributes()
    
    def _set_attributes(self):
        """Set configuration attributes for easier access."""
        # Experiment
        self.exp_name = self.config['experiment']['name']
        self.seed = self.config['experiment']['seed']
        self.output_dir = Path(self.config['experiment']['output_dir'])
        
        # Data
        self.data_root = self.config['data']['root_path']
        self.val_split = self.config['data']['val_split']
        self.batch_size = self.config['data']['batch_size']
        self.num_workers = self.config['data']['num_workers']
        self.image_size = self.config['data']['image_size']
        
        # Model
        self.encoder_name = self.config['model']['encoder']['name']
        self.encoder_weights = self.config['model']['encoder']['pretrained']
        self.use_deep_supervision = self.config['model']['heads']['segmentation']['use_deep_supervision']
        self.separate_detection_fpn = self.config['model']['decoder']['separate_detection_fpn']
        
        # Training
        self.num_epochs = self.config['training']['num_epochs']
        self.learning_rate = self.config['training']['optimizer']['learning_rate']
        self.weight_decay = self.config['training']['optimizer']['weight_decay']
        self.print_freq = self.config['training']['print_freq']
        
        # Device
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Determine device to use for training."""
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by nested key path.
        
        Args:
            key: Dot-separated key path (e.g., 'model.encoder.name')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_task_configs(self) -> List[Dict]:
        """Get all task configurations."""
        return self.config['tasks']

    def set_task_configs_from_dataset(self, task_configs: List[Dict]):
        """Set task configurations from loaded dataset and mark runtime source."""
        self.config['tasks'] = task_configs
        runtime_cfg = self.config.setdefault('runtime', {})
        runtime_cfg['tasks_from_dataset'] = True

    def tasks_from_dataset(self) -> bool:
        """Whether current task configurations were populated from dataset metadata."""
        return bool(self.get('runtime.tasks_from_dataset', False))
    
    def get_loss_config(self, task_name: str) -> Dict:
        """Get loss configuration for specific task."""
        return self.config['training']['loss_configs'].get(task_name, {})
    
    def get_augmentation_config(self, split: str = 'train') -> Dict:
        """Get augmentation configuration for train/val."""
        if split == 'train':
            return self.config['data']['augmentation']['train']
        else:
            return {}
    
    def save(self, save_path: str):
        """Save current configuration to file."""
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def __repr__(self):
        return f"Config(exp_name={self.exp_name}, encoder={self.encoder_name})"


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config object
    """
    return Config(config_path)
