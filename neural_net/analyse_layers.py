import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn
from matplotlib import pyplot

seaborn.set(style='ticks')

path = r'C:\\Users\\Michał\\PycharmProjects\\dsmum-problem-set-3\\neural_net\\'
all_files = glob.glob(path + "*.csv")
print(all_files)
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df.head())
only_epoch = df[df["epoch"] == 800]
only_epoch = only_epoch[["test_loss", "layers", "nodes_per_layer"]]
print(only_epoch.head())
# only_epoch.plot(x= "nodes_per_layer", y='test_loss', kind ='scatter')

fg = seaborn.FacetGrid(data=only_epoch, hue='nodes_per_layer', aspect=3)
fg.map(plt.scatter, 'layers', 'test_loss').add_legend()

plt.show()
