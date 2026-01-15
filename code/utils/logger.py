"""
Training logger for recording metrics during training.
Saves metrics in CSV and JSON formats for easy analysis and plotting.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class TrainingLogger:
    """
    Logger for recording training and validation metrics.
    Saves data in both CSV and JSON formats for easy access.
    """
    
    def __init__(self, log_dir, experiment_name):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.experiment_dir = self.log_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.history = {
            'metadata': {
                'experiment_name': experiment_name,
                'start_time': datetime.now().isoformat(),
                'timestamp': self.timestamp
            },
            'epochs': []
        }
        
        # CSV files
        self.train_loss_csv = self.experiment_dir / 'train_losses.csv'
        self.val_metrics_csv = self.experiment_dir / 'val_metrics.csv'
        self.summary_csv = self.experiment_dir / 'training_summary.csv'
        
        # JSON file
        self.history_json = self.experiment_dir / 'training_history.json'
        
        print(f"\n{'='*80}")
        print(f"Training Logger Initialized")
        print(f"Log directory: {self.experiment_dir}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*80}\n")
    
    def log_epoch(self, epoch, train_losses, val_results_df, learning_rate, epoch_time=None, adaptive_weights=None):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Current epoch number
            train_losses: Dict of {task_id: [losses]} from training
            val_results_df: DataFrame with validation results
            learning_rate: Current learning rate
            epoch_time: Time taken for this epoch (in seconds)
            adaptive_weights: Optional dict with adaptive loss weights and sigmas
        """
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
            'train_losses': {},
            'val_metrics': {}
        }
        
        # Add adaptive weights if provided
        if adaptive_weights is not None:
            epoch_data['adaptive_weights'] = adaptive_weights
        
        # Process training losses
        for task_id, losses in train_losses.items():
            avg_loss = float(np.mean(losses))
            epoch_data['train_losses'][task_id] = {
                'mean': avg_loss,
                'std': float(np.std(losses)),
                'min': float(np.min(losses)),
                'max': float(np.max(losses))
            }
        
        # Process validation metrics
        if not val_results_df.empty:
            for _, row in val_results_df.iterrows():
                task_id = row['Task ID']
                task_metrics = {}
                for col in val_results_df.columns:
                    if col not in ['Task ID', 'Task Name']:
                        task_metrics[col] = float(row[col]) if pd.notna(row[col]) else None
                epoch_data['val_metrics'][task_id] = {
                    'task_name': row['Task Name'],
                    'metrics': task_metrics
                }
        
        # Add to history
        self.history['epochs'].append(epoch_data)
        
        # Save after each epoch
        self._save_all()
    
    def _save_all(self):
        """Save all log files."""
        self._save_json()
        self._save_train_losses_csv()
        self._save_val_metrics_csv()
        self._save_summary_csv()
    
    def _save_json(self):
        """Save complete history as JSON."""
        with open(self.history_json, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _save_train_losses_csv(self):
        """Save training losses in CSV format (easy for plotting)."""
        if not self.history['epochs']:
            return
        
        # Collect all task IDs
        all_tasks = set()
        for epoch_data in self.history['epochs']:
            all_tasks.update(epoch_data['train_losses'].keys())
        all_tasks = sorted(all_tasks)
        
        # Write CSV
        with open(self.train_loss_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['epoch', 'timestamp', 'learning_rate']
            for task in all_tasks:
                fieldnames.extend([
                    f'{task}_loss_mean',
                    f'{task}_loss_std',
                    f'{task}_loss_min',
                    f'{task}_loss_max'
                ])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for epoch_data in self.history['epochs']:
                row = {
                    'epoch': epoch_data['epoch'],
                    'timestamp': epoch_data['timestamp'],
                    'learning_rate': epoch_data['learning_rate']
                }
                for task in all_tasks:
                    if task in epoch_data['train_losses']:
                        loss_data = epoch_data['train_losses'][task]
                        row[f'{task}_loss_mean'] = loss_data['mean']
                        row[f'{task}_loss_std'] = loss_data['std']
                        row[f'{task}_loss_min'] = loss_data['min']
                        row[f'{task}_loss_max'] = loss_data['max']
                    else:
                        row[f'{task}_loss_mean'] = None
                        row[f'{task}_loss_std'] = None
                        row[f'{task}_loss_min'] = None
                        row[f'{task}_loss_max'] = None
                
                writer.writerow(row)
    
    def _save_val_metrics_csv(self):
        """Save validation metrics in CSV format."""
        if not self.history['epochs']:
            return
        
        rows = []
        for epoch_data in self.history['epochs']:
            epoch = epoch_data['epoch']
            timestamp = epoch_data['timestamp']
            
            for task_id, task_data in epoch_data['val_metrics'].items():
                row = {
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'task_id': task_id,
                    'task_name': task_data['task_name']
                }
                row.update(task_data['metrics'])
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.val_metrics_csv, index=False, encoding='utf-8')
    
    def _save_summary_csv(self):
        """Save training summary (one row per epoch with key metrics)."""
        if not self.history['epochs']:
            return
        
        rows = []
        for epoch_data in self.history['epochs']:
            row = {
                'epoch': epoch_data['epoch'],
                'timestamp': epoch_data['timestamp'],
                'learning_rate': epoch_data['learning_rate'],
                'epoch_time': epoch_data.get('epoch_time')
            }
            
            # Average train loss across all tasks
            if epoch_data['train_losses']:
                all_losses = [data['mean'] for data in epoch_data['train_losses'].values()]
                row['avg_train_loss'] = np.mean(all_losses)
            
            # Collect validation metrics
            if epoch_data['val_metrics']:
                all_accuracies = []
                all_f1_scores = []
                all_dice_scores = []
                all_ious = []
                all_maes = []
                
                for task_data in epoch_data['val_metrics'].values():
                    metrics = task_data['metrics']
                    if 'Accuracy' in metrics and metrics['Accuracy'] is not None:
                        all_accuracies.append(metrics['Accuracy'])
                    if 'F1-Score' in metrics and metrics['F1-Score'] is not None:
                        all_f1_scores.append(metrics['F1-Score'])
                    if 'Dice' in metrics and metrics['Dice'] is not None:
                        all_dice_scores.append(metrics['Dice'])
                    if 'IoU' in metrics and metrics['IoU'] is not None:
                        all_ious.append(metrics['IoU'])
                    if 'MAE (pixels)' in metrics and metrics['MAE (pixels)'] is not None:
                        all_maes.append(metrics['MAE (pixels)'])
                
                row['avg_accuracy'] = np.mean(all_accuracies) if all_accuracies else None
                row['avg_f1_score'] = np.mean(all_f1_scores) if all_f1_scores else None
                row['avg_dice'] = np.mean(all_dice_scores) if all_dice_scores else None
                row['avg_iou'] = np.mean(all_ious) if all_ious else None
                row['avg_mae'] = np.mean(all_maes) if all_maes else None
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.summary_csv, index=False, encoding='utf-8')
    
    def save_config(self, config_dict):
        """Save training configuration."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def save_final_summary(self, best_epoch, best_score):
        """Save final training summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.history['metadata']['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_epochs': len(self.history['epochs']),
            'best_epoch': best_epoch,
            'best_validation_score': best_score,
            'timestamp': self.timestamp
        }
        
        summary_path = self.experiment_dir / 'final_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Also save as text for easy reading
        summary_txt = self.experiment_dir / 'final_summary.txt'
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Training Summary - {self.experiment_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Start Time: {summary['start_time']}\n")
            f.write(f"End Time: {summary['end_time']}\n")
            f.write(f"Total Epochs: {summary['total_epochs']}\n")
            f.write(f"Best Epoch: {summary['best_epoch']}\n")
            f.write(f"Best Validation Score: {summary['best_validation_score']:.4f}\n")
            f.write(f"\nLog Directory: {self.experiment_dir}\n")
            f.write(f"\nGenerated Files:\n")
            f.write(f"  - training_history.json (complete history)\n")
            f.write(f"  - train_losses.csv (training losses per epoch)\n")
            f.write(f"  - val_metrics.csv (validation metrics per task per epoch)\n")
            f.write(f"  - training_summary.csv (summary metrics per epoch)\n")
            f.write(f"  - config.json (training configuration)\n")
        
        print(f"\n{'='*80}")
        print(f"Training logs saved to: {self.experiment_dir}")
        print(f"{'='*80}\n")
    
    def get_experiment_dir(self):
        """Get the experiment directory path."""
        return self.experiment_dir


def load_training_history(log_dir):
    """
    Load training history from JSON file.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        dict: Training history
    """
    log_path = Path(log_dir) / 'training_history.json'
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training curves from CSV files.
    
    Args:
        log_dir: Path to the log directory
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
    except ImportError:
        print("matplotlib and seaborn are required for plotting")
        return
    
    log_dir = Path(log_dir)
    
    # Load data
    train_df = pd.read_csv(log_dir / 'train_losses.csv')
    val_df = pd.read_csv(log_dir / 'val_metrics.csv')
    summary_df = pd.read_csv(log_dir / 'training_summary.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Average training loss
    ax1 = axes[0, 0]
    ax1.plot(summary_df['epoch'], summary_df['avg_train_loss'], marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Training Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot 2: Validation metrics
    ax2 = axes[0, 1]
    if 'avg_accuracy' in summary_df.columns:
        ax2.plot(summary_df['epoch'], summary_df['avg_accuracy'], marker='s', label='Accuracy', linewidth=2)
    if 'avg_f1_score' in summary_df.columns:
        ax2.plot(summary_df['epoch'], summary_df['avg_f1_score'], marker='^', label='F1-Score', linewidth=2)
    if 'avg_dice' in summary_df.columns:
        ax2.plot(summary_df['epoch'], summary_df['avg_dice'], marker='d', label='Dice', linewidth=2)
    if 'avg_iou' in summary_df.columns:
        ax2.plot(summary_df['epoch'], summary_df['avg_iou'], marker='v', label='IoU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Learning rate
    ax3 = axes[1, 0]
    ax3.plot(summary_df['epoch'], summary_df['learning_rate'], marker='o', color='green', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Plot 4: Epoch time
    ax4 = axes[1, 1]
    if 'epoch_time' in summary_df.columns and summary_df['epoch_time'].notna().any():
        ax4.plot(summary_df['epoch'], summary_df['epoch_time'], marker='o', color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Epoch Duration')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Epoch Duration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {log_dir / 'training_curves.png'}")
    
    plt.close()


def plot_comprehensive_training_curves(log_dir, save_path=None):
    """
    Plot comprehensive training curves with per-task and average metrics.
    Shows both training and validation metrics for all 4 tasks plus overall average.
    
    Args:
        log_dir: Path to the log directory
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
    except ImportError:
        print("matplotlib and seaborn are required for plotting")
        return
    
    log_dir = Path(log_dir)
    
    # Load data
    train_df = pd.read_csv(log_dir / 'train_losses.csv')
    val_df = pd.read_csv(log_dir / 'val_metrics.csv')
    summary_df = pd.read_csv(log_dir / 'training_summary.csv')
    history = load_training_history(log_dir)
    
    # Task names mapping
    task_names = {
        1: 'Classification',
        2: 'Segmentation', 
        3: 'Keypoint',
        4: 'Measurement'
    }
    
    # Colors for tasks
    task_colors = {
        1: '#1f77b4',  # Blue
        2: '#ff7f0e',  # Orange
        3: '#2ca02c',  # Green
        4: '#d62728',  # Red
    }
    avg_color = '#9467bd'  # Purple for average
    
    # ==================== Figure 1: Training Losses ====================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Training Loss - Per Task & Average', fontsize=16, fontweight='bold')
    
    # Extract per-task training losses
    epochs = summary_df['epoch'].values
    task_train_losses = {task_id: [] for task_id in [1, 2, 3, 4]}
    
    for epoch_data in history['epochs']:
        for task_id in [1, 2, 3, 4]:
            task_key = str(task_id)
            if task_key in epoch_data['train_losses']:
                task_train_losses[task_id].append(epoch_data['train_losses'][task_key]['mean'])
            else:
                task_train_losses[task_id].append(np.nan)
    
    # Plot each task's training loss
    for idx, task_id in enumerate([1, 2, 3, 4]):
        ax = axes1[idx // 3, idx % 3]
        losses = task_train_losses[task_id]
        ax.plot(epochs[:len(losses)], losses, marker='o', linewidth=2, 
                color=task_colors[task_id], markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Task {task_id}: {task_names[task_id]} - Train Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot average training loss
    ax_avg = axes1[1, 1]
    ax_avg.plot(epochs, summary_df['avg_train_loss'], marker='o', linewidth=2, 
                color=avg_color, markersize=4, label='Average')
    ax_avg.set_xlabel('Epoch')
    ax_avg.set_ylabel('Loss')
    ax_avg.set_title('Average Training Loss (All Tasks)')
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend()
    
    # Plot all tasks together for comparison
    ax_all = axes1[1, 2]
    for task_id in [1, 2, 3, 4]:
        losses = task_train_losses[task_id]
        ax_all.plot(epochs[:len(losses)], losses, marker='o', linewidth=2, 
                    color=task_colors[task_id], markersize=3, 
                    label=f'Task {task_id}: {task_names[task_id]}')
    ax_all.plot(epochs, summary_df['avg_train_loss'], marker='s', linewidth=2.5, 
                color=avg_color, markersize=4, label='Average', linestyle='--')
    ax_all.set_xlabel('Epoch')
    ax_all.set_ylabel('Loss')
    ax_all.set_title('All Tasks Training Loss Comparison')
    ax_all.legend(loc='upper right', fontsize=8)
    ax_all.grid(True, alpha=0.3)
    
    plt.tight_layout()
    train_loss_path = save_path.replace('.png', '_train_loss.png') if save_path else log_dir / 'training_loss_per_task.png'
    plt.savefig(train_loss_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {train_loss_path}")
    plt.close()
    
    # ==================== Figure 2: Validation Metrics ====================
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Validation Metrics - Per Task & Average', fontsize=16, fontweight='bold')
    
    # Extract per-task validation metrics
    task_val_metrics = {task_id: {'primary': [], 'epochs': []} for task_id in [1, 2, 3, 4]}
    
    # Define primary metric for each task type
    # Task 1: Classification -> Accuracy/F1
    # Task 2: Segmentation -> Dice/IoU
    # Task 3: Keypoint -> MAE (lower is better, so we'll use negative or handle separately)
    # Task 4: Measurement -> MAE
    
    for epoch_data in history['epochs']:
        epoch = epoch_data['epoch']
        for task_id in [1, 2, 3, 4]:
            task_key = str(task_id)
            if task_key in epoch_data['val_metrics']:
                metrics = epoch_data['val_metrics'][task_key]['metrics']
                task_val_metrics[task_id]['epochs'].append(epoch)
                
                # Get primary metric based on task type
                if task_id == 1:  # Classification
                    val = metrics.get('F1-Score') or metrics.get('Accuracy')
                elif task_id == 2:  # Segmentation
                    val = metrics.get('Dice') or metrics.get('IoU')
                elif task_id in [3, 4]:  # Keypoint/Measurement - use negative MAE for consistent plotting
                    mae = metrics.get('MAE (pixels)') or metrics.get('MAE')
                    val = mae  # Keep as is, will handle in plotting
                else:
                    val = None
                task_val_metrics[task_id]['primary'].append(val)
    
    # Plot each task's validation metric
    metric_names = {
        1: 'F1-Score / Accuracy',
        2: 'Dice / IoU',
        3: 'MAE (pixels) ↓',
        4: 'MAE (pixels) ↓'
    }
    
    for idx, task_id in enumerate([1, 2, 3, 4]):
        ax = axes2[idx // 3, idx % 3]
        epochs_t = task_val_metrics[task_id]['epochs']
        values = task_val_metrics[task_id]['primary']
        
        if epochs_t and values:
            ax.plot(epochs_t, values, marker='s', linewidth=2, 
                    color=task_colors[task_id], markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_names[task_id])
        ax.set_title(f'Task {task_id}: {task_names[task_id]} - Val {metric_names[task_id]}')
        ax.grid(True, alpha=0.3)
        
        # For MAE tasks, indicate lower is better
        if task_id in [3, 4]:
            ax.invert_yaxis()  # Invert so lower MAE appears higher (better)
    
    # Plot average validation metrics
    ax_avg2 = axes2[1, 1]
    if 'avg_f1_score' in summary_df.columns:
        ax_avg2.plot(epochs, summary_df['avg_f1_score'], marker='s', linewidth=2, 
                     label='Avg F1-Score', color='#1f77b4')
    if 'avg_dice' in summary_df.columns:
        ax_avg2.plot(epochs, summary_df['avg_dice'], marker='^', linewidth=2, 
                     label='Avg Dice', color='#ff7f0e')
    if 'avg_accuracy' in summary_df.columns:
        ax_avg2.plot(epochs, summary_df['avg_accuracy'], marker='o', linewidth=2, 
                     label='Avg Accuracy', color='#2ca02c')
    ax_avg2.set_xlabel('Epoch')
    ax_avg2.set_ylabel('Score')
    ax_avg2.set_title('Average Validation Metrics')
    ax_avg2.legend(loc='lower right', fontsize=8)
    ax_avg2.grid(True, alpha=0.3)
    
    # Plot combined view - normalized scores
    ax_combined = axes2[1, 2]
    for task_id in [1, 2]:  # Higher is better tasks
        epochs_t = task_val_metrics[task_id]['epochs']
        values = task_val_metrics[task_id]['primary']
        if epochs_t and values:
            ax_combined.plot(epochs_t, values, marker='o', linewidth=2, 
                            color=task_colors[task_id], markersize=3,
                            label=f'Task {task_id}: {task_names[task_id]}')
    ax_combined.set_xlabel('Epoch')
    ax_combined.set_ylabel('Score (higher is better)')
    ax_combined.set_title('Classification & Segmentation Metrics')
    ax_combined.legend(loc='lower right', fontsize=8)
    ax_combined.grid(True, alpha=0.3)
    
    plt.tight_layout()
    val_metrics_path = save_path.replace('.png', '_val_metrics.png') if save_path else log_dir / 'validation_metrics_per_task.png'
    plt.savefig(val_metrics_path, dpi=300, bbox_inches='tight')
    print(f"Validation metrics plot saved to: {val_metrics_path}")
    plt.close()
    
    # ==================== Figure 3: Combined Train & Val Summary ====================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Training & Validation Summary - All Tasks', fontsize=16, fontweight='bold')
    
    # Plot 1: All tasks training loss
    ax1 = axes3[0, 0]
    for task_id in [1, 2, 3, 4]:
        losses = task_train_losses[task_id]
        ax1.plot(epochs[:len(losses)], losses, marker='o', linewidth=1.5, 
                color=task_colors[task_id], markersize=3, alpha=0.7,
                label=f'T{task_id}: {task_names[task_id]}')
    ax1.plot(epochs, summary_df['avg_train_loss'], marker='s', linewidth=2.5, 
            color=avg_color, markersize=4, label='Average', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss - All Tasks')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classification & Segmentation val metrics (higher is better)
    ax2 = axes3[0, 1]
    for task_id in [1, 2]:
        epochs_t = task_val_metrics[task_id]['epochs']
        values = task_val_metrics[task_id]['primary']
        if epochs_t and values:
            ax2.plot(epochs_t, values, marker='s', linewidth=2, 
                    color=task_colors[task_id], markersize=4,
                    label=f'T{task_id}: {task_names[task_id]}')
    if 'avg_f1_score' in summary_df.columns and 'avg_dice' in summary_df.columns:
        avg_score = (summary_df['avg_f1_score'].fillna(0) + summary_df['avg_dice'].fillna(0)) / 2
        ax2.plot(epochs, avg_score, marker='D', linewidth=2, 
                color=avg_color, markersize=4, linestyle='--', label='Avg (F1+Dice)/2')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score (↑ better)')
    ax2.set_title('Validation: Classification & Segmentation')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Keypoint & Measurement val metrics (lower is better)
    ax3 = axes3[1, 0]
    for task_id in [3, 4]:
        epochs_t = task_val_metrics[task_id]['epochs']
        values = task_val_metrics[task_id]['primary']
        if epochs_t and values:
            ax3.plot(epochs_t, values, marker='s', linewidth=2, 
                    color=task_colors[task_id], markersize=4,
                    label=f'T{task_id}: {task_names[task_id]}')
    if 'avg_mae' in summary_df.columns:
        ax3.plot(epochs, summary_df['avg_mae'], marker='D', linewidth=2, 
                color=avg_color, markersize=4, linestyle='--', label='Average MAE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE (pixels) (↓ better)')
    ax3.set_title('Validation: Keypoint & Measurement')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate & epoch time
    ax4 = axes3[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(epochs, summary_df['learning_rate'], marker='o', linewidth=2, 
                     color='#17becf', markersize=3, label='Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate', color='#17becf')
    ax4.set_yscale('log')
    ax4.tick_params(axis='y', labelcolor='#17becf')
    
    if 'epoch_time' in summary_df.columns and summary_df['epoch_time'].notna().any():
        line2 = ax4_twin.plot(epochs, summary_df['epoch_time'], marker='s', linewidth=2, 
                              color='#bcbd22', markersize=3, label='Epoch Time')
        ax4_twin.set_ylabel('Epoch Time (s)', color='#bcbd22')
        ax4_twin.tick_params(axis='y', labelcolor='#bcbd22')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=8)
    else:
        ax4.legend(loc='upper right', fontsize=8)
    
    ax4.set_title('Learning Rate & Training Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = save_path if save_path else log_dir / 'training_summary_comprehensive.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive summary plot saved to: {summary_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Generated plots:")
    print(f"  1. {train_loss_path}")
    print(f"  2. {val_metrics_path}")
    print(f"  3. {summary_path}")
    print(f"{'='*60}\n")


__all__ = ['TrainingLogger', 'load_training_history', 'plot_training_curves', 'plot_comprehensive_training_curves']
