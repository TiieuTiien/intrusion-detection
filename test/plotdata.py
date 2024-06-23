# Plot data
import matplotlib.pyplot as plt
import numpy as np

tn, fp, fn, tp = 14160, 8, 11, 15524

TPR = tp / (tp + fn) * 100
TNR = tn / (tn + fp) * 100
# Bar chart
bar_labels = ['Correct', 'Incorrect']

colors = ['tab:green', 'tab:orange']
counts = [tp, tn]

# Displaying true positive and true negative rates as text on the plot
p = plt.bar(bar_labels, counts, color=colors, label=None)
plt.xlabel('Classification')
plt.ylabel('Count')
plt.title('Correct and Incorrect Classifications')

# Displaying true positive and true negative rates as text on the plot
plt.bar_label(p, labels = [f'{TPR:.2f}%', f'{TNR:.2f}%'], label_type='center', color='white')
 
plt.title('Correct and Incorrect Classifications')


plt.show()