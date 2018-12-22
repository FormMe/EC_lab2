import pandas as pd
import matplotlib.pyplot as plt

from TSPGeneticAlgo import TSPGeneticAlgo, City

data = pd.read_excel('data.xlsx')
points = [City(i[1].x, i[1].y) for i in data.iterrows()]

algo = TSPGeneticAlgo(points, 50, 10, 0.3)
algo.eval(10000)
# e.draw_path(e.pop_ranked[0][0])
# solution
#
# xs = [i.x for i in self.pop[individ]]
# ys = [i.y for i in self.pop[individ]]
# plt.plot(xs, ys, '-o')

plt.plot(algo.history)
