import matplotlib.pyplot as plt
import numpy as np

def create_boxplot(data, labels, title):
	fig, ax = plt.subplots()

	# Group three boxes per plot
	num_boxes = len(data)
	group_size = 1
	num_plots = num_boxes // group_size
	plot_indices = [i * group_size for i in range(num_plots)]

	# Calculate the width of each group and the spacing between groups
	width = 0.4
	spacing = 0.2
	total_width = num_plots * (group_size * width + spacing) - spacing
	plt.title(title)
	for i, index in enumerate(plot_indices):
		subset_data = data[index:index+group_size]
		subset_labels = labels[index:index+group_size]

		# Calculate the x-coordinates for each boxplot within the group
		positions = np.arange(index+1, index+group_size+1) + i * (group_size * width + spacing) - total_width / 2

		# Plot the boxplot
		bp = ax.boxplot(subset_data, labels=subset_labels, positions=positions, widths=width)

		# Annotate the minimum and maximum values of boxes and whiskers
		for j, box in enumerate(bp['boxes']):
			x = positions[j]
			y_min = box.get_ydata().min()
			y_max = box.get_ydata().max()
			whisker_min = bp['whiskers'][j*2].get_ydata().min()
			whisker_max = bp['whiskers'][j*2 + 1].get_ydata().max()

			#ax.annotate(f'{y_min:.1f}', xy=(x, y_min), xytext=(10, -10), textcoords='offset points', va='top')
			#ax.annotate(f'{y_max:.1f}', xy=(x, y_max), xytext=(10, 10), textcoords='offset points', va='bottom')
			#ax.annotate(f'{whisker_min:.1f}', xy=(x, whisker_min), xytext=(10, 15), textcoords='offset points', va='top')
			#ax.annotate(f'{whisker_max:.1f}', xy=(x, whisker_max), xytext=(10, -15), textcoords='offset points', va='bottom')
	plt.savefig(title+".jpg")
	plt.show()

# Example usage
values = []

boxplot_labels = []

title = ""

with open("box_config.txt") as box_f:
	lines = box_f.read().splitlines();
	title = lines[0]
	for line in lines[1:]:
		print(line)
		l = line.split("|");
		boxplot_labels.append(l[0])
		data = []
		with open(l[1]+".csv") as data_f:
			for data_l in data_f.read().splitlines()[1:]:
					data.append(float(data_l.split(",")[int(l[2])]))
		values.append(data)


create_boxplot(values, boxplot_labels, title)
