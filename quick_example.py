"""
快速示例：如何使用训练日志进行分析

这个脚本展示了最常用的日志分析功能。
适合快速查看训练结果和生成论文图表。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def quick_analysis(log_dir):
    """
    快速分析训练结果
    
    Args:
        log_dir: 训练日志目录路径
    """
    log_dir = Path(log_dir)
    
    print("\n" + "="*80)
    print(f"分析训练日志: {log_dir.name}")
    print("="*80)
    
    # 1. 读取数据
    summary_df = pd.read_csv(log_dir / 'training_summary.csv')
    
    # 2. 显示基本信息
    print(f"\n总训练轮数: {len(summary_df)}")
    print(f"\n最后一轮结果:")
    last_epoch = summary_df.iloc[-1]
    print(f"  - 训练损失: {last_epoch['avg_train_loss']:.4f}")
    if 'avg_accuracy' in summary_df.columns and pd.notna(last_epoch.get('avg_accuracy')):
        print(f"  - 准确率: {last_epoch['avg_accuracy']:.4f}")
    if 'avg_f1_score' in summary_df.columns and pd.notna(last_epoch.get('avg_f1_score')):
        print(f"  - F1分数: {last_epoch['avg_f1_score']:.4f}")
    if 'avg_dice' in summary_df.columns and pd.notna(last_epoch.get('avg_dice')):
        print(f"  - Dice系数: {last_epoch['avg_dice']:.4f}")
    if 'avg_iou' in summary_df.columns and pd.notna(last_epoch.get('avg_iou')):
        print(f"  - IoU: {last_epoch['avg_iou']:.4f}")
    
    # 3. 找出最佳结果
    print(f"\n最佳结果:")
    print(f"  - 最低损失: {summary_df['avg_train_loss'].min():.4f} (Epoch {summary_df['avg_train_loss'].idxmin() + 1})")
    if 'avg_accuracy' in summary_df.columns and summary_df['avg_accuracy'].notna().any():
        best_acc_idx = summary_df['avg_accuracy'].idxmax()
        print(f"  - 最高准确率: {summary_df.loc[best_acc_idx, 'avg_accuracy']:.4f} (Epoch {best_acc_idx + 1})")
    
    # 4. 生成快速可视化
    print(f"\n生成可视化图表...")
    output_dir = log_dir / 'quick_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # 图1: 训练损失和主要指标
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 训练损失
    axes[0].plot(summary_df['epoch'], summary_df['avg_train_loss'], 
                marker='o', linewidth=2, markersize=5, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Average Training Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 验证指标
    ax = axes[1]
    plotted = False
    if 'avg_accuracy' in summary_df.columns and summary_df['avg_accuracy'].notna().any():
        ax.plot(summary_df['epoch'], summary_df['avg_accuracy'], 
               marker='s', linewidth=2, markersize=5, label='Accuracy')
        plotted = True
    if 'avg_f1_score' in summary_df.columns and summary_df['avg_f1_score'].notna().any():
        ax.plot(summary_df['epoch'], summary_df['avg_f1_score'], 
               marker='^', linewidth=2, markersize=5, label='F1-Score')
        plotted = True
    if 'avg_dice' in summary_df.columns and summary_df['avg_dice'].notna().any():
        ax.plot(summary_df['epoch'], summary_df['avg_dice'], 
               marker='d', linewidth=2, markersize=5, label='Dice')
        plotted = True
    if 'avg_iou' in summary_df.columns and summary_df['avg_iou'].notna().any():
        ax.plot(summary_df['epoch'], summary_df['avg_iou'], 
               marker='v', linewidth=2, markersize=5, label='IoU')
        plotted = True
    
    if plotted:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_overview.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {save_path}")
    plt.close()
    
    # 图2: 学习率变化
    plt.figure(figsize=(10, 5))
    plt.plot(summary_df['epoch'], summary_df['learning_rate'], 
            marker='o', linewidth=2, markersize=5, color='#2ecc71')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / 'learning_rate.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {save_path}")
    plt.close()
    
    # 5. 保存关键数据到文本文件
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("训练结果摘要\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验: {log_dir.name}\n")
        f.write(f"总轮数: {len(summary_df)}\n\n")
        
        f.write("最后一轮结果:\n")
        f.write(f"  训练损失: {last_epoch['avg_train_loss']:.4f}\n")
        if 'avg_accuracy' in summary_df.columns and pd.notna(last_epoch.get('avg_accuracy')):
            f.write(f"  准确率: {last_epoch['avg_accuracy']:.4f}\n")
        if 'avg_f1_score' in summary_df.columns and pd.notna(last_epoch.get('avg_f1_score')):
            f.write(f"  F1分数: {last_epoch['avg_f1_score']:.4f}\n")
        if 'avg_dice' in summary_df.columns and pd.notna(last_epoch.get('avg_dice')):
            f.write(f"  Dice系数: {last_epoch['avg_dice']:.4f}\n")
        if 'avg_iou' in summary_df.columns and pd.notna(last_epoch.get('avg_iou')):
            f.write(f"  IoU: {last_epoch['avg_iou']:.4f}\n")
        
        f.write(f"\n最佳结果:\n")
        f.write(f"  最低损失: {summary_df['avg_train_loss'].min():.4f} (Epoch {summary_df['avg_train_loss'].idxmin() + 1})\n")
        if 'avg_accuracy' in summary_df.columns and summary_df['avg_accuracy'].notna().any():
            best_acc_idx = summary_df['avg_accuracy'].idxmax()
            f.write(f"  最高准确率: {summary_df.loc[best_acc_idx, 'avg_accuracy']:.4f} (Epoch {best_acc_idx + 1})\n")
    
    print(f"  ✓ 保存: {summary_file}")
    
    print(f"\n{'='*80}")
    print(f"分析完成！结果保存在: {output_dir}")
    print(f"{'='*80}\n")


def compare_experiments(log_dirs, metric='avg_accuracy', output_path='comparison.png'):
    """
    对比多个实验的结果
    
    Args:
        log_dirs: 日志目录列表
        metric: 要对比的指标
        output_path: 输出图片路径
    """
    plt.figure(figsize=(12, 6))
    
    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        df = pd.read_csv(log_dir / 'training_summary.csv')
        
        if metric in df.columns:
            plt.plot(df['epoch'], df[metric], 
                    marker='o', linewidth=2, markersize=5, 
                    label=log_dir.name)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.title(f'{metric.replace("_", " ").title()} Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图保存到: {output_path}")
    plt.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  快速分析单个实验:")
        print("    python quick_example.py <日志目录>")
        print("\n  对比多个实验:")
        print("    python quick_example.py <日志目录1> <日志目录2> ... --compare")
        print("\n示例:")
        print("    python quick_example.py outputs/experiment1_20260101_123456")
        print("    python quick_example.py outputs/exp1 outputs/exp2 outputs/exp3 --compare")
        sys.exit(1)
    
    if '--compare' in sys.argv:
        # 对比模式
        log_dirs = [arg for arg in sys.argv[1:] if arg != '--compare']
        if len(log_dirs) < 2:
            print("错误：对比模式至少需要2个日志目录")
            sys.exit(1)
        
        print(f"\n对比 {len(log_dirs)} 个实验...")
        compare_experiments(log_dirs, metric='avg_accuracy', output_path='accuracy_comparison.png')
        compare_experiments(log_dirs, metric='avg_train_loss', output_path='loss_comparison.png')
        print("\n对比完成！")
    else:
        # 单个分析模式
        log_dir = sys.argv[1]
        quick_analysis(log_dir)
