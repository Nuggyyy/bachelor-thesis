import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from the LaTeX table
data = {
    'Qwen3-ASR': {
        'LoRA': {'English': 6.8, 'Irish': 70.8, 'Shona': 71.3, 'Javanese': 35.2},
        'VeRA': {'English': 9.9, 'Irish': 73.7, 'Shona': 77.9, 'Javanese': 41.9},
        'RandLoRA': {'English': 6.75, 'Irish': 60.7, 'Shona': 51.4, 'Javanese': 27.6},
    },
    'Whisper': {
        'LoRA': {'English': 6.2, 'Irish': 77.25, 'Shona': 65.7, 'Javanese': 39.1},
        'VeRA': {'English': 6.2, 'Irish': 80.5, 'Shona': 98.25, 'Javanese': 85.9},
        'RandLoRA': {'English': 12.2, 'Irish': 62.8, 'Shona': 43.3, 'Javanese': 29.7},
    },
    'w2v-BERT 2.0': {
        'LoRA': {'English': 23.7, 'Irish': 66.2, 'Shona': 23.4, 'Javanese': 19.4},
        'VeRA': {'English': 58.4, 'Irish': 83.1, 'Shona': 41.4, 'Javanese': 36.3},
        'RandLoRA': {'English': 20.7, 'Irish': 57.9, 'Shona': 23.8, 'Javanese': 19.1},
    }
}

# Get unique PEFT methods and languages
peft_methods = ['LoRA', 'VeRA', 'RandLoRA']
languages = ['English', 'Irish', 'Shona', 'Javanese']
models = ['Qwen3-ASR', 'Whisper', 'w2v-BERT 2.0']

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(16, 6))

# Prepare data for plotting
bar_width = 0.25
x_positions = []
x_labels = []
current_x = 0

colors = {'Qwen3-ASR': '#1f77b4', 'Whisper': '#ff7f0e', 'w2v-BERT 2.0': '#2ca02c'}

# For each PEFT method
for peft_idx, peft in enumerate(peft_methods):
    # For each language
    for lang_idx, lang in enumerate(languages):
        # Collect values for this language and PEFT method across models
        values = []
        available_models = []
        
        for model in models:
            if peft in data[model] and lang in data[model][peft]:
                values.append(data[model][peft][lang])
                available_models.append(model)
        
        # Plot bars for each model
        num_models = len(available_models)
        offset = (num_models - 1) * bar_width / 2
        
        for model_idx, (model, value) in enumerate(zip(available_models, values)):
            bar_x = current_x + (model_idx * bar_width) - offset
            ax.bar(bar_x, value, bar_width, label=model if peft_idx == 0 and lang_idx == 0 else "",
                   color=colors[model], alpha=0.8)
            x_positions.append(bar_x)
        
        # Center label for this language group
        x_labels.append(lang)
        current_x += 1.5
    
    # Add spacing between PEFT methods
    current_x += 0.5

# Create proper x-axis labels
unique_x = []
unique_labels = []
current_x = 0
for peft_idx, peft in enumerate(peft_methods):
    for lang_idx, lang in enumerate(languages):
        unique_x.append(current_x)
        unique_labels.append(lang)
        current_x += 1.5
    current_x += 0.5

ax.set_xticks(unique_x)
ax.set_xticklabels(unique_labels, fontsize=10)

# Add PEFT method labels at the top
peft_positions = []
current_x = 0
for peft_idx, peft in enumerate(peft_methods):
    peft_start = current_x
    peft_end = current_x + (4 * 1.5) - 0.5  # 4 languages, 1.5 spacing each
    peft_center = (peft_start + peft_end) / 2
    peft_positions.append((peft_center, peft))
    current_x = peft_end + 0.5

# Add PEFT method labels as text at the top
for pos, peft in peft_positions:
    ax.text(pos, ax.get_ylim()[1] * 0.95, peft, ha='center', fontsize=11, fontweight='bold')

# Add vertical separators between PEFT methods
current_x = 0
for peft_idx, peft in enumerate(peft_methods):
    current_x += (4 * 1.5) - 0.5
    if peft_idx < len(peft_methods) - 1:
        ax.axvline(x=current_x + 0.25, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    current_x += 0.5

# Labels and title
ax.set_ylabel('WER (Word Error Rate)', fontsize=12, fontweight='bold')
ax.set_title('WER Comparison per PEFT Method and Language', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Create custom legend (only show each model once)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[model], alpha=0.8, label=model) for model in models]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Set y-axis to start at 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('/home/serafin/repositories/bachelor-thesis/local/wer_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to local/wer_plot.png")
plt.show()
