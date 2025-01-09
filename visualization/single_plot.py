import matplotlib.pyplot as plt

# Enable LaTeX text rendering for bold labels
plt.rc('text', usetex=True)

# Model names used in the experiments
models = [
    'GCN', 'GCN + \nfiltering', 'GIN', 'GIN + \nfiltering', 
    r'\textbf{DeltaGNN} \neigen', r'\textbf{DeltaGNN} \ndegree', r'\textbf{DeltaGNN} \ncurvature'
]

# Accuracy and standard deviation for Organ-S dataset
organs_acc = [60.12, 58.98, 60.36, 61.51, 62.66, 62.64, 63.06]
organs_std = [0.08, 0.41, 0.00, 0.47, 0.24, 0.22, 0.42]

# Accuracy and standard deviation for Organ-C dataset
organc_acc = [77.68, 77.15, 76.33, 77.26, 80.22, 79.73, 80.27]
organc_std = [0.35, 0.29, 0.00, 1.47, 1.14, 1.05, 0.53]

# Define colors for the plots
color_s = '#F8A2A2'  # Color for Organ-S
color_c = '#8670FD'  # Color for Organ-C

# Calculate maximum on accuracies
max_organs_acc = max(organs_acc)
max_organc_acc = max(organc_acc)

# Plot for CiteSeer dataset
plt.figure(figsize=(12, 6))
plt.errorbar(models, organs_acc, yerr=organs_std, fmt='o-', color=color_s, ecolor='black', capsize=2, label='Organ-S')
plt.xlabel('Methods', fontsize=14)  # X-axis label
plt.ylabel('Accuracy (%)', fontsize=14)  # Y-axis label
plt.title('MedMNIST Organ-S Accuracy with Standard Deviation', fontsize=16)  # Plot title
plt.xticks(fontsize=12, rotation=45)  # X-axis ticks
plt.yticks(fontsize=12)  # Y-axis ticks
plt.ylim(min(organs_acc)-1,max(organs_acc)+1)

# Annotate the bars with their values
for i, v in enumerate(organs_acc):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold' if v == max_organs_acc else 'normal')

plt.grid(True)  # Show grid
plt.legend()  # Show legend
plt.tight_layout()  # Adjust layout
plt.savefig('organ_s_accuracy.png', transparent=True)
plt.show()  # Display the plot

# Plot for PubMed dataset
plt.figure(figsize=(12, 6))
plt.errorbar(models, organc_acc, yerr=organc_std, fmt='o-', color=color_c, ecolor='black', capsize=2, label='Organ-C')
plt.xlabel('Methods', fontsize=14)  # X-axis label
plt.ylabel('Accuracy (%)', fontsize=14)  # Y-axis label
plt.title('MedMNIST Organ-C Accuracy with Standard Deviation', fontsize=16)  # Plot title
plt.xticks(fontsize=12, rotation=45)  # X-axis ticks
plt.yticks(fontsize=12)  # Y-axis ticks
plt.ylim(min(organc_acc)-1.5,max(organc_acc)+1.5)

# Annotate the bars with their values
for i, v in enumerate(organc_acc):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold' if v == max_organc_acc else 'normal')

plt.grid(True)  # Show grid
plt.legend()  # Show legend
plt.tight_layout()  # Adjust layout
plt.savefig('organ_c_accuracy.png', transparent=True)
plt.show()  # Display the plot
