import matplotlib.pyplot as plt
import pandas as pd

plot1_dframe = pd.read_csv("dataset_median_mad1_st0.95_it0.95_l2_over0.34.csv",delimiter="|",header=0)
plot1_dframe.plot(x="dist_a",y="dist_c",kind='hexbin',xlim=(20,60),ylim=(25,100),grid=True,gridsize=60,figsize=(5,5))
plt.tight_layout()
plt.savefig('Plot_all_2layer.png')
plt.show()