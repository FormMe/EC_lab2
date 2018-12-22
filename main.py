import pandas as pd
import matplotlib.pyplot as plt

from TSPGeneticAlgo import TSPGeneticAlgo, City

data = pd.read_excel('data.xlsx')
points = [City(i[1].x, i[1].y) for i in data.iterrows()]

algo = TSPGeneticAlgo(points, 100, 20, 1e-5)
algo.eval(5000)

xs, ys = algo.get_path()
plt.plot(xs, ys, '-o')
plt.show()

plt.ylabel("Path length")
plt.xlabel("Generation")
plt.plot(algo.history)
plt.show()
