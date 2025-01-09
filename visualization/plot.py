import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
methods = ['GCN', 'GIN', 'GBK-GNN', 'DeltaGNN random', 'DeltaGNN eigen', 'DeltaGNN degree', 'DeltaGNN curvature']
cite_seer_acc = [75.60, 73.12, 79.18, 79.40, 80.10, 79.30, 79.20]
pub_med_acc = [87.40, 85.76, 89.11, 90.25, 89.85, 89.77, 0.0]  # Placeholder for DeltaGNN curvature

# Standard deviations for the accuracy measurements
cite_seer_std = [0.29, 1.61, 0.96, 0.65, 0.68, 0.24, 0.29]
pub_med_std = [0.18, 0.29, 0.23, 0.52, 0.40, 0.46, 0.0]  # Placeholder for DeltaGNN curvature

# Define positions and width for the bars
x = np.arange(len(methods))  # Positions of the labels on the x-axis
width = 0.25  # Width of the bars
model_space = 0.5  # Space between different model groups

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars for CiteSeer accuracy with error bars for standard deviation
rects2 = ax.bar(x, cite_seer_acc, width, yerr=cite_seer_std, label='CiteSeer', color='#F8A2A2', capsize=5)

# Plot bars for PubMed accuracy with error bars for standard deviation
rects3 = ax.bar(x + width, pub_med_acc, width, yerr=pub_med_std, label='PubMed', color='#BEFCB6', capsize=5)

# Set y-axis limits
ax.set_ylim(50, 100)

# Add labels, title, and legend
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Prediction Result Comparison of DeltaGNN and SOTA Methods', fontsize=16)
ax.set_xticks(x + width / 2)  # Center x-ticks between groups of bars
ax.set_xticklabels(methods, fontsize=12)
ax.legend(fontsize=12)  # Font size for legend

def autolabel(rects, dataset_acc):
    """
    Annotate bars with their height values.

    Args:
        rects (list of Rectangle): List of bar rectangles to annotate.
        dataset_acc (list of float): List of accuracy values corresponding to the bars.
    """
    # Find the maximum value and its index
    max_idx = np.argmax(dataset_acc)
    max_value = dataset_acc[max_idx]
    
    for i, rect in enumerate(rects):
        height = rect.get_height()
        
        # Determine the vertical offset for the labels
        vertical_offset = 17
        
        # Bold annotate the maximum value
        if height == max_value:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, vertical_offset),  # Offset from the bar
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10,  # Font size for the label
                        weight='bold')  # Make the font bold
        else:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, vertical_offset),  # Offset from the bar
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)  # Regular font for other bars

# Annotate bars with their values
autolabel(rects2, cite_seer_acc)
autolabel(rects3, pub_med_acc)

# Adjust layout for a clean fit
fig.tight_layout()

# Display the plot
plt.show()
