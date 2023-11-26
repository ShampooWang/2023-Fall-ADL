import matplotlib.pyplot as plt

# Assuming perplexities is your list of perplexity values
perplexities = [3.941579797744751, 3.720828266143799, 3.603021628379822, 3.549933352470398, 3.4714290714263916, 3.4433025484085085, 3.4349288964271545, 
                3.426289901256561, 3.367016493797302, 3.3945897126197817, 3.3432593784332276, 3.365777750492096,
                3.524483560085297, 3.617946572780609, 3.524483560085297, 3.617946572780609, 3.529282047748566, 3.552972758293152, 3.5800789017677306, 3.450218511581421]

# Generate steps (assuming 100 steps)
steps = list(range(0, len(perplexities) * 100, 100))

# Plotting
plt.plot(steps, perplexities, marker='o', linestyle='-', color='b')
plt.title('Avg. perplexity on public test set')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.savefig("ppl.png")
