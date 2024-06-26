import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_data(dataset_file, model_file):
	try:
		dataset = pd.read_csv(dataset_file)
		model = pd.read_csv(model_file)

		a = model.loc[0, 'a']
		b = model.loc[0, 'b']

		plt.scatter(dataset['km'], dataset['price'], marker='o', label='Data')

		x_values = dataset['km']
		y_values = a * x_values + b
		plt.plot(x_values, y_values, color='red', label=f'Linear regression')

		plt.grid(True)
		plt.xlabel('km')
		plt.ylabel('price')
		plt.legend()

		plt.show()
	except Exception as e:
		print(f"Error: {e}")
		exit(1)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python plot.py <dataset_file> <model_file>")
		sys.exit(1)

	dataset_file = sys.argv[1]
	model_file = sys.argv[2]
	plot_data(dataset_file, model_file)
