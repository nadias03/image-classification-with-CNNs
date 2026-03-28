import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
from matplotlib.lines import Line2D

PALETTE = {
    'train':  '#2E86AB',
    'valid':  '#E84855',
    'bg':     '#FFFFFF',
    'grid':   '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['bg'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        PALETTE['grid'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.7,
    'font.family':       'sans-serif',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.frameon':    False,
})


def summarize_results(results, hyperparameter='lr', model_name="Model"):
    rows = []
    for param, seeds_dict in results.items():
        valid_accs = [v["valid_acc"] for v in seeds_dict.values()]
        valid_losses = [v["valid_loss"] for v in seeds_dict.values()]
        test_accs  = [v["test_acc"]  for v in seeds_dict.values()]
        test_losses = [v["test_loss"] for v in seeds_dict.values()]

        rows.append({
            "Model":           model_name,
            f"{hyperparameter}":   param,
            "Valid Acc Mean":  np.mean(valid_accs),
            "Valid Acc Std":   np.std(valid_accs),
            "Valid Loss Mean":  np.mean(valid_losses),
            "Valid Loss Std":   np.std(valid_losses),
            "Test Acc Mean":   np.mean(test_accs),
            "Test Acc Std":    np.std(test_accs),
            "Test Loss Mean":  np.mean(test_losses),
            "Test Loss Std":   np.std(test_losses),
        })
    return pd.DataFrame(rows)


def print_summary_table(df, hyperparameter):
    display_df = df.copy()
    display_df["Valid Acc"]  = display_df.apply(
        lambda r: f"{r['Valid Acc Mean']:.4f} ± {r['Valid Acc Std']:.4f}", axis=1)
    display_df["Valid Loss"]  = display_df.apply(
        lambda r: f"{r['Valid Loss Mean']:.4f} ± {r['Valid Loss Std']:.4f}", axis=1)
    display_df["Test Acc"]   = display_df.apply(
        lambda r: f"{r['Test Acc Mean']:.4f} ± {r['Test Acc Std']:.4f}", axis=1)
    display_df["Test Loss"]  = display_df.apply(
        lambda r: f"{r['Test Loss Mean']:.4f} ± {r['Test Loss Std']:.4f}", axis=1)

    cols = ["Model", f"{hyperparameter}", "Valid Acc", "Valid Loss", "Test Acc", "Test Loss"]
    return display_df[cols]

def plot_loss_comparison(results, param_name, param_name_short):
    """
    Plot mean train/validation loss over epochs for each value of tested hyperparameter,
    with ±1 std shading across seeds.

    Args:
        results (dict): Output of function for experiments.
                        Structure: results[param_name][seed]['history']
                        where history has 'train_loss' and 'valid_loss'.
        param_name (str): Name of the hyperparameter that was tested in experiment.
        param_name_short (str): Short name of the hyperparameter that was tested in experiment, to be displayed on legened
    """

    params = list(results.keys())
    colors = sns.color_palette("deep", n_colors=len(params))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f'Loss per {param_name} (mean ± std across seeds)',
                 fontsize=15, fontweight='bold', y=1.02)

    for idx, (param, seed_dict) in enumerate(results.items()):
        color = colors[idx]
        label = f'{param_name}={param}'

        train_losses = np.array([seed_dict[s]['history']['train_loss'] for s in seed_dict])
        valid_losses = np.array([seed_dict[s]['history']['valid_loss'] for s in seed_dict])

        epochs = range(1, train_losses.shape[1] + 1)

        train_mean, train_std = train_losses.mean(axis=0), train_losses.std(axis=0)
        valid_mean, valid_std = valid_losses.mean(axis=0), valid_losses.std(axis=0)


        ax1.plot(epochs, train_mean, color=color, linewidth=2, label=label)
        ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.6)

        ax2.plot(epochs, valid_mean, color=color, linewidth=2, linestyle='--', label=label)
        ax2.fill_between(epochs, valid_mean - valid_std, valid_mean + valid_std,
                         color=color, alpha=0.2)
        
        ax1.set_title('Training Loss',   fontsize=14, fontweight='bold')
        ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')

    for ax in (ax1, ax2):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss',  fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    ax2.tick_params(labelleft=True)

    legend_handles = [Line2D([0], [0], color=colors[i], linewidth=2, label=f'{param_name_short}={lr}')
                      for i, lr in enumerate(params)]
    
    fig.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(params),
               fontsize=10,
               title=f'{param_name}')

    fig.suptitle(f'Loss per {param_name} (mean ± std across seeds)',
                 fontsize=15, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(results, param_name, param_name_short):
    """
    Plot mean accuracy for validation over epochs for each value of tested hyperparameter,
    with ±1 std shading across seeds and mean accuracy computed on test set.

    Args:
        results (dict): Output of function for experiments.
                        Structure: results[param_name][seed]['history']
                        where history has 'train_loss' and 'valid_loss'.
        param_name (str): Name of the hyperparameter that was tested in experiment.
        param_name_short (str): Short name of the hyperparameter that was tested in experiment, to be displayed on legened
    """
    params = list(results.keys())
    colors = sns.color_palette("deep", n_colors=len(params))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for idx, (param, seed_dict) in enumerate(results.items()):
        color = colors[idx]

        valid_accs = np.array([seed_dict[s]['history']['valid_acc'] for s in seed_dict])
        epochs = range(1, valid_accs.shape[1] + 1)

        mean = valid_accs.mean(axis=0)
        std = valid_accs.std(axis=0)

        best_epoch = int(np.argmax(mean)) + 1
        best_acc = mean[best_epoch - 1]

        ax1.plot(epochs, mean, color=color, linewidth=2, label=f'{param_name_short}={param}')
        ax1.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax1.scatter([best_epoch], [best_acc], color=color, zorder=5)
        ax1.annotate(f'{best_acc:.3f}',
                     xy=(best_epoch, best_acc),
                     xytext=(6, 6), textcoords='offset points',
                     fontsize=8, color=color)

    ax1.set_title(f'Validation Accuracy per {param_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    means = [np.mean([v['test_acc'] for v in results[param].values()]) for param in params]
    stds = [np.std( [v['test_acc'] for v in results[param].values()]) for param in params]
    x = np.arange(len(params))

    bars = ax2.bar(x, means, yerr=stds, capsize=5,
                   color=colors, edgecolor='black', linewidth=0.7)
    ax2.set_title('Test Accuracy (mean ± std)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(param) for param in params], rotation=30)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.06))
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.set_xlabel(f'{param_name}', fontsize=12)

    fig.align_xlabels([ax1, ax2])

    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.003,
                 f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()