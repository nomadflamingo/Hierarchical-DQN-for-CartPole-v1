path = "lumi_results.log"
import matplotlib.pyplot as plt

# Load the data from the text file
data = []
with open(path, 'r') as file:
    for line in file:
        line_data = line.split()
        data.append(float(line_data[3]))

# Generate x values
x = list(range(len(data)))

# Plot the data
plt.plot(data)
plt.title('Reward function values')
plt.show()
