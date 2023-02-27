import matplotlib.pyplot as plt

x = ['70%', '80%', '90%']
original = [0.96, 0.96, 0.96]
pruned_un = [0.9566, 0.946, 0.8845]
finetuned_un = [0.9682, 0.9656, 0.9615]

pruned = [0.1982, 0.1938, 0.1219]
finetuned = [0.3972, 0.3102, 0.2004]

plt.plot(x, original, label='original')
plt.plot(x, pruned_un, label='pruned_un')
plt.plot(x, finetuned_un, label='finetuned_un')
plt.plot(x, pruned, label='pruned')
plt.plot(x, finetuned, label='finetuned')

# Add a legend
plt.legend()

# Add labels to the x and y axes
plt.xlabel('Percentage of Sparsity')
plt.ylabel('Accuracy')

# Show the plot
plt.show()
