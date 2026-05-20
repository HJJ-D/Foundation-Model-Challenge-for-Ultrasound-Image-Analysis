"""
Training logger for recording metrics during training.
Saves metrics in CSV and JSON formats for easy analysis and plotting.
"""

import json
import csv
import yaml
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
                'timestamp': self.timestamp,
                'model_params_total': None,
                'model_params_trainable': None,
                'model_param_groups': None,
            },
            'epochs': []
        }
        
        # CSV files
        self.train_loss_csv = self.experiment_dir / 'train_losses.csv'
        self.val_metrics_csv = self.experiment_dir / 'val_metrics.csv'
        self.summary_csv = self.experiment_dir / 'training_summary.csv'
        self.moe_stats_csv = self.experiment_dir / 'moe_stats.csv'
        self.runtime_txt = self.experiment_dir / 'model_runtime_summary.txt'
        
        # JSON file
        self.history_json = self.experiment_dir / 'training_history.json'
        
        print(f"\n{'='*80}")
        print(f"Training Logger Initialized")
        print(f"Log directory: {self.experiment_dir}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*80}\n")
    
    def log_epoch(
        self,
        epoch,
        train_losses,
        val_results_df,
        learning_rate,
        epoch_time=None,
        train_time=None,
        inference_time=None,
        adaptive_weights=None,
        moe_stats=None
    ):
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
            'train_time': train_time,
            'inference_time': inference_time,
            'train_losses': {},
            'val_metrics': {}
        }
        
        # Add adaptive weights if provided
        if adaptive_weights is not None:
            epoch_data['adaptive_weights'] = adaptive_weights
        if moe_stats is not None:
            epoch_data['moe_stats'] = moe_stats
        
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
        # Also save a human-readable validation summary (per-epoch)
        self._save_summary_csv()
        self._save_moe_stats_csv()
        self._save_runtime_summary_txt()
    
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
                'epoch_time': epoch_data.get('epoch_time'),
                'train_time': epoch_data.get('train_time'),
                'inference_time': epoch_data.get('inference_time')
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
                all_mae_stds = []

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
                    if 'MAE_std (pixels)' in metrics and metrics['MAE_std (pixels)'] is not None:
                        all_mae_stds.append(metrics['MAE_std (pixels)'])

                row['avg_accuracy'] = np.mean(all_accuracies) if all_accuracies else None
                row['avg_f1_score'] = np.mean(all_f1_scores) if all_f1_scores else None
                row['avg_dice'] = np.mean(all_dice_scores) if all_dice_scores else None
                row['avg_iou'] = np.mean(all_ious) if all_ious else None
                row['avg_mae'] = np.mean(all_maes) if all_maes else None
                row['avg_mae_std'] = np.mean(all_mae_stds) if all_mae_stds else None
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.summary_csv, index=False, encoding='utf-8')

    def _save_moe_stats_csv(self):
        """Save MoE importance/load stats per task and per task group."""
        if not self.history['epochs']:
            return

        max_experts = 0
        for epoch_data in self.history['epochs']:
            moe_stats = epoch_data.get('moe_stats', {})
            for scope_key in ('by_task_id', 'by_task_name'):
                for entry in moe_stats.get(scope_key, {}).values():
                    importance = entry.get('importance', [])
                    if len(importance) > max_experts:
                        max_experts = len(importance)

        if max_experts == 0:
            return

        rows = []
        for epoch_data in self.history['epochs']:
            epoch = epoch_data['epoch']
            timestamp = epoch_data['timestamp']
            moe_stats = epoch_data.get('moe_stats', {})
            for scope_key, scope_name in (('by_task_id', 'task_id'), ('by_task_name', 'task_name')):
                entries = moe_stats.get(scope_key, {})
                for key, entry in entries.items():
                    row = {
                        'epoch': epoch,
                        'timestamp': timestamp,
                        'scope': scope_name,
                        'id': key,
                        'task_name': entry.get('task_name'),
                        'aux_loss': entry.get('aux_loss')
                    }
                    importance = entry.get('importance', [])
                    load = entry.get('load', [])
                    for i in range(max_experts):
                        row[f'importance_{i}'] = importance[i] if i < len(importance) else None
                        row[f'load_{i}'] = load[i] if i < len(load) else None
                    rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.moe_stats_csv, index=False, encoding='utf-8')

    def set_model_stats(self, total_params, trainable_params=None, groups=None):
        """Record model parameter statistics.

        Args:
            total_params: Total parameter count (sum of p.numel() over all params).
            trainable_params: Subset where requires_grad=True.
            groups: Optional dict from utils.summarize_parameter_groups, mapping
                group name -> {'total': int, 'trainable': int}. When provided,
                a per-group breakdown is written into model_runtime_summary.txt.
        """
        total_params = int(total_params) if total_params is not None else None
        trainable_params = int(trainable_params) if trainable_params is not None else None
        self.history['metadata']['model_params_total'] = total_params
        self.history['metadata']['model_params_trainable'] = trainable_params
        if groups is not None:
            normalized = {}
            for name, stats in groups.items():
                normalized[str(name)] = {
                    'total': int(stats.get('total', 0) or 0),
                    'trainable': int(stats.get('trainable', 0) or 0),
                }
            self.history['metadata']['model_param_groups'] = normalized
        self._save_runtime_summary_txt()
        self._save_json()

    @staticmethod
    def _format_seconds(seconds):
        if seconds is None:
            return "N/A"
        return f"{float(seconds):.2f}s"

    def _get_runtime_stats(self):
        """Aggregate runtime statistics from epoch history."""
        epoch_times = []
        train_times = []
        inference_times = []

        for epoch_data in self.history.get('epochs', []):
            epoch_time = epoch_data.get('epoch_time')
            train_time = epoch_data.get('train_time')
            inference_time = epoch_data.get('inference_time')
            if epoch_time is not None:
                epoch_times.append(float(epoch_time))
            if train_time is not None:
                train_times.append(float(train_time))
            if inference_time is not None:
                inference_times.append(float(inference_time))

        def _safe_mean(values):
            return float(np.mean(values)) if values else None

        return {
            'total_epoch_time': float(np.sum(epoch_times)) if epoch_times else None,
            'total_train_time': float(np.sum(train_times)) if train_times else None,
            'total_inference_time': float(np.sum(inference_times)) if inference_times else None,
            'avg_epoch_time': _safe_mean(epoch_times),
            'avg_train_time': _safe_mean(train_times),
            'avg_inference_time': _safe_mean(inference_times),
            'num_epochs_with_time': len(epoch_times),
            'num_epochs_with_train_time': len(train_times),
            'num_epochs_with_inference_time': len(inference_times),
        }

    def _save_runtime_summary_txt(self):
        """Save model parameter count and runtime statistics to txt."""
        metadata = self.history.get('metadata', {})
        runtime = self._get_runtime_stats()
        total_params = metadata.get('model_params_total')
        trainable_params = metadata.get('model_params_trainable')
        non_trainable_params = (
            total_params - trainable_params
            if total_params is not None and trainable_params is not None
            else None
        )

        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"Model and Runtime Summary - {self.experiment_name}")
        lines.append(f"{'='*80}")
        lines.append(f"Generated At: {datetime.now().isoformat()}")
        lines.append("")
        lines.append("Model Parameters:")
        lines.append(
            f"  - Total Parameters: {f'{int(total_params):,}' if total_params is not None else 'N/A'}"
        )
        lines.append(
            f"  - Trainable Parameters: {f'{int(trainable_params):,}' if trainable_params is not None else 'N/A'}"
        )
        lines.append(
            f"  - Non-trainable Parameters: {f'{int(non_trainable_params):,}' if non_trainable_params is not None else 'N/A'}"
        )

        param_groups = metadata.get('model_param_groups')
        if param_groups:
            lines.append("")
            lines.append("Parameter Breakdown (by group):")
            header = f"  {'Group':<20s} {'Total':>15s} {'Trainable':>15s}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            sum_total = 0
            sum_trainable = 0
            for group_name, stats in param_groups.items():
                g_total = int(stats.get('total', 0) or 0)
                g_trainable = int(stats.get('trainable', 0) or 0)
                sum_total += g_total
                sum_trainable += g_trainable
                if g_total == 0:
                    continue  # hide empty groups to keep the table compact
                lines.append(
                    f"  {group_name:<20s} {g_total:>15,} {g_trainable:>15,}"
                )
            lines.append("  " + "-" * (len(header) - 2))
            lines.append(
                f"  {'TOTAL':<20s} {sum_total:>15,} {sum_trainable:>15,}"
            )

        lines.append("")
        lines.append("Runtime:")
        lines.append(f"  - Total Training Time: {self._format_seconds(runtime.get('total_train_time'))}")
        lines.append(f"  - Total Inference Time: {self._format_seconds(runtime.get('total_inference_time'))}")
        lines.append(f"  - Total Epoch Time: {self._format_seconds(runtime.get('total_epoch_time'))}")
        lines.append(f"  - Avg Training Time / Epoch: {self._format_seconds(runtime.get('avg_train_time'))}")
        lines.append(f"  - Avg Inference Time / Epoch: {self._format_seconds(runtime.get('avg_inference_time'))}")
        lines.append(f"  - Avg Epoch Time: {self._format_seconds(runtime.get('avg_epoch_time'))}")
        lines.append(
            f"  - Epochs with Timing: train={runtime.get('num_epochs_with_train_time', 0)}, "
            f"inference={runtime.get('num_epochs_with_inference_time', 0)}, "
            f"epoch_total={runtime.get('num_epochs_with_time', 0)}"
        )

        try:
            with open(self.runtime_txt, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + "\n")
        except Exception as e:
            print(f"Could not write {self.runtime_txt.name}: {e}")

    def _save_best_model_summary_txt(self, best_model_eval_on_train=None, best_epoch=None):
        """Save a human-readable validation summary for best epoch (and last epoch for comparison).

        This writes `best_model_summary.txt` in the experiment directory containing:
          - Per-task validation metrics for the best epoch
          - Optional per-task validation metrics for the last epoch (comparison)
          - Mean primary metric for the four high-level task groups:
            classification, segmentation, detection, regression
          - Best model evaluation on the training set (if provided)
        """
        if not self.history['epochs']:
            return

        def _append_epoch_section(lines_ref, epoch_data, section_title):
            if 'val_metrics' not in epoch_data or not epoch_data['val_metrics']:
                lines_ref.append(f"{section_title}: N/A (no validation metrics)")
                lines_ref.append("")
                return

            lines_ref.append(section_title)
            lines_ref.append(f"Epoch: {epoch_data.get('epoch', 'N/A')}")
            lines_ref.append(f"Timestamp: {epoch_data.get('timestamp', 'N/A')}")
            lines_ref.append("")
            lines_ref.append("Per-task validation metrics:")
            lines_ref.append("")

            group_names = ['classification', 'segmentation', 'detection', 'regression']
            group_vals = {g: [] for g in group_names}
            regression_stds = []   # per-task MAE_std values for the regression group
            classification_metrics = {'Accuracy': [], 'F1-Score': []}

            def _first_present(metrics_dict, keys):
                for key in keys:
                    if key in metrics_dict and metrics_dict[key] is not None:
                        return metrics_dict[key]
                return None

            for task_id in sorted(epoch_data['val_metrics'].keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                task_data = epoch_data['val_metrics'][task_id]
                task_name = task_data.get('task_name', '')
                metrics = task_data.get('metrics', {})

                metric_parts = []
                for k, v in metrics.items():
                    if v is None:
                        metric_parts.append(f"{k}: N/A")
                    else:
                        try:
                            metric_parts.append(f"{k}: {float(v):.4f}")
                        except Exception:
                            metric_parts.append(f"{k}: {v}")

                lines_ref.append(f"  - Task {task_id} | {task_name} -> " + ", ".join(metric_parts))

                tn = str(task_name).lower()
                for g in group_names:
                    if g in tn:
                        if g == 'classification':
                            acc = metrics.get('Accuracy')
                            f1 = metrics.get('F1-Score')
                            if acc is not None:
                                try:
                                    classification_metrics['Accuracy'].append(float(acc))
                                except Exception:
                                    pass
                            if f1 is not None:
                                try:
                                    classification_metrics['F1-Score'].append(float(f1))
                                except Exception:
                                    pass
                        elif g == 'segmentation':
                            val = _first_present(metrics, ['Dice', 'IoU'])
                            if val is not None:
                                try:
                                    group_vals[g].append(float(val))
                                except Exception:
                                    pass
                        elif g == 'detection':
                            val = metrics.get('IoU')
                            if val is not None:
                                try:
                                    group_vals[g].append(float(val))
                                except Exception:
                                    pass
                        else:  # regression
                            val = _first_present(metrics, ['MAE', 'MAE (pixels)'])
                            if val is not None:
                                try:
                                    group_vals[g].append(float(val))
                                except Exception:
                                    pass
                            std_val = metrics.get('MAE_std (pixels)')
                            if std_val is not None:
                                try:
                                    regression_stds.append(float(std_val))
                                except Exception:
                                    pass

            lines_ref.append("")
            lines_ref.append("Group mean primary metrics:")
            acc_vals = classification_metrics['Accuracy']
            f1_vals = classification_metrics['F1-Score']
            if acc_vals:
                mean_acc = float(np.mean(acc_vals))
                lines_ref.append(f"  - Classification Accuracy: {mean_acc:.4f} (mean over {len(acc_vals)} task(s))")
            else:
                lines_ref.append("  - Classification Accuracy: N/A (no tasks found)")
            if f1_vals:
                mean_f1 = float(np.mean(f1_vals))
                lines_ref.append(f"  - Classification F1-Score: {mean_f1:.4f} (mean over {len(f1_vals)} task(s))")
            else:
                lines_ref.append("  - Classification F1-Score: N/A (no tasks found)")

            for g in ['segmentation', 'detection']:
                vals = group_vals[g]
                if vals:
                    mean_val = float(np.mean(vals))
                    lines_ref.append(f"  - {g.title()}: {mean_val:.4f} (mean over {len(vals)} task(s))")
                else:
                    lines_ref.append(f"  - {g.title()}: N/A (no tasks found)")

            # Regression: show MAE ± STD
            reg_vals = group_vals['regression']
            if reg_vals:
                mean_mae = float(np.mean(reg_vals))
                if regression_stds:
                    mean_std = float(np.mean(regression_stds))
                    lines_ref.append(
                        f"  - Regression: MAE={mean_mae:.4f} ± STD={mean_std:.4f}"
                        f" (mean over {len(reg_vals)} task(s))"
                    )
                else:
                    lines_ref.append(f"  - Regression: MAE={mean_mae:.4f} (mean over {len(reg_vals)} task(s))")
            else:
                lines_ref.append("  - Regression: N/A (no tasks found)")
            lines_ref.append("")

        # Resolve best epoch entry from history
        best_epoch_data = None
        used_best_epoch_fallback = False
        if best_epoch is not None:
            for e in self.history['epochs']:
                if e.get('epoch') == best_epoch:
                    best_epoch_data = e
                    break
        if best_epoch_data is None:
            best_epoch_data = self.history['epochs'][-1]
            used_best_epoch_fallback = (best_epoch is not None)

        last_epoch_data = self.history['epochs'][-1]

        lines = []
        lines.append("Validation Summary")
        if used_best_epoch_fallback:
            lines.append(
                f"Warning: requested best_epoch={best_epoch} not found in history; "
                "falling back to last epoch metrics."
            )
        lines.append("")
        _append_epoch_section(lines, best_epoch_data, "Best Epoch Validation Metrics")

        if last_epoch_data.get('epoch') != best_epoch_data.get('epoch'):
            _append_epoch_section(lines, last_epoch_data, "Last Epoch Validation Metrics (for comparison)")

        # Add best model evaluation on training set if provided
        if best_model_eval_on_train:
            lines.append("Best Model Evaluation on Training Set:")
            for task_group, score in best_model_eval_on_train.items():
                if isinstance(score, dict):
                    if task_group == 'regression':
                        mae = score.get("MAE (pixels)")
                        std = score.get("MAE_std (pixels)")
                        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
                        std_str = f" ± {std:.4f}" if std is not None else ""
                        lines.append(f"  - {task_group.title()}: MAE={mae_str}{std_str}")
                    else:
                        acc = score.get("Accuracy")
                        f1 = score.get("F1-Score")
                        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                        lines.append(f"  - {task_group.title()}: Accuracy={acc_str}, F1-Score={f1_str}")
                elif score is not None:
                    lines.append(f"  - {task_group.title()}: {score:.4f}")
                else:
                    lines.append(f"  - {task_group.title()}: N/A")

        # Write to file
        out_path = self.experiment_dir / 'best_model_summary.txt'
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                for l in lines:
                    f.write(l + "\n")
        except Exception as e:
            print(f"Could not write best_model_summary.txt: {e}")
    
    def save_config(self, config_dict):
        """Save training configuration."""
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def save_final_summary(self, best_epoch, best_score):
        """Save final training summary."""
        runtime = self._get_runtime_stats()
        metadata = self.history.get('metadata', {})
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.history['metadata']['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_epochs': len(self.history['epochs']),
            'best_epoch': best_epoch,
            'best_validation_score': best_score,
            'model_params_total': metadata.get('model_params_total'),
            'model_params_trainable': metadata.get('model_params_trainable'),
            'total_train_time': runtime.get('total_train_time'),
            'total_inference_time': runtime.get('total_inference_time'),
            'total_epoch_time': runtime.get('total_epoch_time'),
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
            model_total = summary.get('model_params_total')
            model_trainable = summary.get('model_params_trainable')
            model_non_trainable = (
                model_total - model_trainable
                if model_total is not None and model_trainable is not None
                else None
            )
            f.write(
                f"Model Parameters (Total): {f'{int(model_total):,}' if model_total is not None else 'N/A'}\n"
            )
            f.write(
                f"Model Parameters (Trainable): {f'{int(model_trainable):,}' if model_trainable is not None else 'N/A'}\n"
            )
            f.write(
                f"Model Parameters (Non-trainable): {f'{int(model_non_trainable):,}' if model_non_trainable is not None else 'N/A'}\n"
            )
            f.write(f"Total Training Time: {self._format_seconds(summary.get('total_train_time'))}\n")
            f.write(f"Total Inference Time: {self._format_seconds(summary.get('total_inference_time'))}\n")
            f.write(f"Total Epoch Time: {self._format_seconds(summary.get('total_epoch_time'))}\n")
            f.write(f"\nLog Directory: {self.experiment_dir}\n")

            # Best-epoch regression summary (MAE ± STD)
            best_epoch_data = next(
                (e for e in self.history['epochs'] if e.get('epoch') == best_epoch), None
            ) or (self.history['epochs'][-1] if self.history['epochs'] else None)
            if best_epoch_data:
                reg_maes, reg_stds = [], []
                for task_data in best_epoch_data.get('val_metrics', {}).values():
                    if 'regression' in str(task_data.get('task_name', '')).lower():
                        m = task_data.get('metrics', {})
                        if m.get('MAE (pixels)') is not None:
                            reg_maes.append(float(m['MAE (pixels)']))
                        if m.get('MAE_std (pixels)') is not None:
                            reg_stds.append(float(m['MAE_std (pixels)']))
                if reg_maes:
                    mean_mae = float(np.mean(reg_maes))
                    std_part = f" ± STD={float(np.mean(reg_stds)):.4f}" if reg_stds else ""
                    f.write(f"Best Epoch Regression (val): MAE={mean_mae:.4f}{std_part}"
                            f" (mean over {len(reg_maes)} task(s))\n")

            f.write(f"\nGenerated Files:\n")
            f.write(f"  - training_history.json (complete history)\n")
            f.write(f"  - train_losses.csv (training losses per epoch)\n")
            f.write(f"  - val_metrics.csv (validation metrics per task per epoch)\n")
            f.write(f"  - training_summary.csv (summary metrics per epoch)\n")
            f.write(f"  - moe_stats.csv (MoE importance/load per task)\n")
            f.write(f"  - model_runtime_summary.txt (model parameters and runtime summary)\n")
            f.write(f"  - config.yaml (training configuration)\n")

        self._save_runtime_summary_txt()
        
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
    if 'avg_mae' in summary_df.columns:
        ax2.plot(summary_df['epoch'], summary_df['avg_mae'], marker='x', label='MAE', linewidth=2, color='tab:orange')
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
    Plot comprehensive training curves with per-group and average metrics.
    Shows both training and validation metrics for classification/segmentation/
    detection/regression groups plus overall average.

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
    val_df = pd.read_csv(log_dir / 'val_metrics.csv')
    summary_df = pd.read_csv(log_dir / 'training_summary.csv')
    history = load_training_history(log_dir)

    if not history.get('epochs'):
        print("No training history found, cannot plot comprehensive curves.")
        return

    # Build task_id -> task_name map from validation metrics
    task_id_to_name = {}
    if not val_df.empty and {'task_id', 'task_name'}.issubset(val_df.columns):
        for _, row in val_df[['task_id', 'task_name']].dropna().drop_duplicates().iterrows():
            task_id_to_name[str(row['task_id'])] = str(row['task_name'])

    if not task_id_to_name:
        for epoch_data in history.get('epochs', []):
            for task_id, task_data in epoch_data.get('val_metrics', {}).items():
                tname = task_data.get('task_name')
                if tname:
                    task_id_to_name[str(task_id)] = str(tname)

    def map_group(task_name):
        if not task_name:
            return None
        tn = str(task_name).lower()
        if 'classification' in tn:
            return 'Classification'
        if 'segmentation' in tn:
            return 'Segmentation'
        if 'detection' in tn:
            return 'Detection'
        if 'regression' in tn:
            return 'Regression'
        return None

    def pick_metric(metrics, keys):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
        return None

    group_order = ['Classification', 'Segmentation', 'Detection', 'Regression']

    # Colors for groups
    task_colors = {
        'Classification': '#1f77b4',  # Blue
        'Segmentation': '#ff7f0e',    # Orange
        'Detection': '#2ca02c',       # Green
        'Regression': '#d62728',      # Red
    }
    avg_color = '#9467bd'  # Purple for average

    # ==================== Figure 1: Training Losses ====================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Training Loss - Per Group & Average', fontsize=16, fontweight='bold')

    # Extract per-group training losses
    epochs = [epoch_data['epoch'] for epoch_data in history['epochs']]
    group_train_losses = {group: [] for group in group_order}

    for epoch_data in history['epochs']:
        group_values = {group: [] for group in group_order}
        for task_id, loss_data in epoch_data.get('train_losses', {}).items():
            group = map_group(task_id_to_name.get(str(task_id), ''))
            if group is None:
                continue
            mean_val = loss_data.get('mean')
            if mean_val is not None:
                group_values[group].append(mean_val)

        for group in group_order:
            if group_values[group]:
                group_train_losses[group].append(float(np.mean(group_values[group])))
            else:
                group_train_losses[group].append(np.nan)

    # Plot each group's training loss
    for idx, group in enumerate(group_order):
        ax = axes1[idx // 3, idx % 3]
        losses = group_train_losses[group]
        ax.plot(epochs[:len(losses)], losses, marker='o', linewidth=2,
                color=task_colors[group], markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{group} - Train Loss')
        ax.grid(True, alpha=0.3)

    # Plot average training loss
    ax_avg = axes1[1, 1]
    avg_train_loss = None
    if 'avg_train_loss' in summary_df.columns:
        avg_train_loss = summary_df['avg_train_loss']
    elif history['epochs']:
        avg_train_loss = [
            np.mean([d['mean'] for d in epoch_data['train_losses'].values()])
            for epoch_data in history['epochs']
            if epoch_data.get('train_losses')
        ]
    if avg_train_loss is not None and len(avg_train_loss) > 0:
        ax_avg.plot(epochs, avg_train_loss, marker='o', linewidth=2,
                    color=avg_color, markersize=4, label='Average')
    ax_avg.set_xlabel('Epoch')
    ax_avg.set_ylabel('Loss')
    ax_avg.set_title('Average Training Loss (All Tasks)')
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend()

    # Plot all groups together for comparison
    ax_all = axes1[1, 2]
    for group in group_order:
        losses = group_train_losses[group]
        ax_all.plot(epochs[:len(losses)], losses, marker='o', linewidth=2,
                    color=task_colors[group], markersize=3,
                    label=group)
    if avg_train_loss is not None and len(avg_train_loss) > 0:
        ax_all.plot(epochs, avg_train_loss, marker='s', linewidth=2.5,
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
    fig2.suptitle('Validation Metrics - Per Group & Average', fontsize=16, fontweight='bold')

    # Extract per-group validation metrics
    group_val_metrics = {group: {'primary': [], 'epochs': []} for group in group_order}

    for epoch_data in history['epochs']:
        epoch = epoch_data['epoch']
        group_values = {group: [] for group in group_order}
        for task_id, task_data in epoch_data.get('val_metrics', {}).items():
            task_name = task_data.get('task_name', '')
            group = map_group(task_name)
            if group is None:
                continue
            metrics = task_data.get('metrics', {})

            if group == 'Classification':
                val = pick_metric(metrics, ['F1-Score', 'Accuracy'])
            elif group == 'Segmentation':
                val = pick_metric(metrics, ['Dice', 'IoU'])
            elif group == 'Detection':
                val = pick_metric(metrics, ['IoU'])
            else:
                val = pick_metric(metrics, ['MAE (pixels)', 'MAE'])

            if val is not None:
                group_values[group].append(val)

        for group in group_order:
            if group_values[group]:
                group_val_metrics[group]['epochs'].append(epoch)
                group_val_metrics[group]['primary'].append(float(np.mean(group_values[group])))

    # Plot each group's validation metric
    metric_names = {
        'Classification': 'F1-Score / Accuracy',
        'Segmentation': 'Dice',
        'Detection': 'IoU',
        'Regression': 'MAE (pixels) (lower is better)'
    }

    for idx, group in enumerate(group_order):
        ax = axes2[idx // 3, idx % 3]
        epochs_t = group_val_metrics[group]['epochs']
        values = group_val_metrics[group]['primary']

        if epochs_t and values:
            ax.plot(epochs_t, values, marker='s', linewidth=2,
                    color=task_colors[group], markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_names[group])
        ax.set_title(f'{group} - Val {metric_names[group]}')
        ax.grid(True, alpha=0.3)

        # For MAE tasks, indicate lower is better
        if group == 'Regression':
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
    if 'avg_iou' in summary_df.columns:
        ax_avg2.plot(epochs, summary_df['avg_iou'], marker='D', linewidth=2,
                     label='Avg IoU', color='#d62728')
    if 'avg_mae' in summary_df.columns:
        ax_avg2.plot(epochs, summary_df['avg_mae'], marker='v', linewidth=2,
                     label='Avg MAE', color='#7f7f7f')
    ax_avg2.set_xlabel('Epoch')
    ax_avg2.set_ylabel('Score')
    ax_avg2.set_title('Average Validation Metrics')
    ax_avg2.legend(loc='lower right', fontsize=8)
    ax_avg2.grid(True, alpha=0.3)

    # Plot combined view - higher is better
    ax_combined = axes2[1, 2]
    for group in ['Classification', 'Segmentation', 'Detection']:
        epochs_t = group_val_metrics[group]['epochs']
        values = group_val_metrics[group]['primary']
        if epochs_t and values:
            ax_combined.plot(epochs_t, values, marker='o', linewidth=2,
                            color=task_colors[group], markersize=3,
                            label=group)
    ax_combined.set_xlabel('Epoch')
    ax_combined.set_ylabel('Score (higher is better)')
    ax_combined.set_title('Classification, Segmentation, Detection Metrics')
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

    # Plot 1: All groups training loss
    ax1 = axes3[0, 0]
    for group in group_order:
        losses = group_train_losses[group]
        ax1.plot(epochs[:len(losses)], losses, marker='o', linewidth=1.5,
                color=task_colors[group], markersize=3, alpha=0.7,
                label=group)
    if avg_train_loss is not None and len(avg_train_loss) > 0:
        ax1.plot(epochs, avg_train_loss, marker='s', linewidth=2.5,
                color=avg_color, markersize=4, label='Average', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss - All Tasks')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Classification, Segmentation, Detection val metrics (higher is better)
    ax2 = axes3[0, 1]
    for group in ['Classification', 'Segmentation', 'Detection']:
        epochs_t = group_val_metrics[group]['epochs']
        values = group_val_metrics[group]['primary']
        if epochs_t and values:
            ax2.plot(epochs_t, values, marker='s', linewidth=2,
                    color=task_colors[group], markersize=4,
                    label=group)
    score_cols = [c for c in ['avg_f1_score', 'avg_accuracy', 'avg_dice', 'avg_iou'] if c in summary_df.columns]
    if score_cols:
        avg_score = summary_df[score_cols].mean(axis=1, skipna=True)
        ax2.plot(epochs, avg_score, marker='D', linewidth=2,
                color=avg_color, markersize=4, linestyle='--', label='Avg (higher is better)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score (higher is better)')
    ax2.set_title('Validation: Classification, Segmentation, Detection')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regression val metrics (lower is better)
    ax3 = axes3[1, 0]
    epochs_t = group_val_metrics['Regression']['epochs']
    values = group_val_metrics['Regression']['primary']
    if epochs_t and values:
        ax3.plot(epochs_t, values, marker='s', linewidth=2,
                color=task_colors['Regression'], markersize=4,
                label='Regression')
    if 'avg_mae' in summary_df.columns:
        ax3.plot(epochs, summary_df['avg_mae'], marker='D', linewidth=2,
                color=avg_color, markersize=4, linestyle='--', label='Average MAE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE (pixels) (lower is better)')
    ax3.set_title('Validation: Regression')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate & epoch time
    ax4 = axes3[1, 1]
    ax4_twin = ax4.twinx()

    if 'learning_rate' in summary_df.columns:
        line1 = ax4.plot(epochs, summary_df['learning_rate'], marker='o', linewidth=2,
                         color='#17becf', markersize=3, label='Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate', color='#17becf')
        ax4.set_yscale('log')
        ax4.tick_params(axis='y', labelcolor='#17becf')
    else:
        line1 = []

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
