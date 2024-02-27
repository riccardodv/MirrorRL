import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np
os.getcwd()
from json import load


# import matplotlib.pyplot as plt
from cycler import cycler
# import numpy as np
from matplotlib.lines import Line2D

files = [
            "errors_pendulum_fqi.npy",
            "test_errors_pendulum_fqi.npy",
            "during_train_pendulum_fqi.npy"
            # "errors_pendulum_lstd.npy",
            # "errors_pendulum_lstd_random.npy"
        ]

# c_cyc = cycler(color = list('rgb')) 
c_cyc = cycler(color = list('rbg')) 
ls_cyc = cycler(linestyle=['solid', 'dashed'])
plt_cycle = c_cyc * ls_cyc * cycler(linewidth=[1.])

print("plt_cycle", plt_cycle)
                        
d_msbe = []
d_mse = []
for f in files:
    data = np.load(f, allow_pickle=True).item()
    if f == "during_train_pendulum_fqi.npy":
        d_msbe.append(data['msbe'][::30])
    else:
        d_msbe.append(data["msbe"])
    if "mse" in data.keys():
        d_mse.append(data["mse"])
    else:
        d_mse.append([])
    # d_mse.append(data["mse"]) 

for i, pmts in enumerate(plt_cycle):
    if i%2 == 0:
        plt.semilogy(d_msbe[i//2], **pmts)
    else:
        plt.semilogy(d_mse[i//2], **pmts)

legend_elements = [Line2D([0], [0], color='r', label='fqi_after_train', linewidth=1.),
                    Line2D([0], [0], color='b', label='fqi_test', linewidth=1.),
                    Line2D([0], [0], color='g', label='fqi_during_train', linewidth=1.),
                    # Line2D([0], [0], color='b', label='lstd', linewidth=1.),
                    # Line2D([0], [0], color='g', label='lstd_random', linewidth=1.)]
]
plt.legend(handles=legend_elements)
plt.savefig("./Figures/comparison_fqi_training&testing_errors++"+".pdf", format='pdf')
