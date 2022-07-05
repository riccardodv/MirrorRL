import pickle
import matplotlib.pyplot as plt
import torch

RATE = 0



weights = pickle.load(open("weights_cos.pkl", "rb"))


def significant_number(w, rate=1e-6):
    feat_num = w.shape[1]
    w_filtered = torch.where(torch.abs(w) > rate, w, torch.zeros_like(w))
    # w_rejected = torch.where(torch.abs(w)<= rate, w, torch.zeros_like(w))
    num_significant = torch.count_nonzero(w_filtered)
    # num_significant = torch.count_nonzero(w)
    # print("Rejected!")
    # print(w_rejected)
    return num_significant.item(), feat_num

what_to_plot = [[*significant_number(w, RATE)] for w in weights]
what_to_plot = torch.tensor(what_to_plot)

print("At the last epoch:")
print(what_to_plot[-1])

print("THE LAST WEIGHTS")
print(weights[-1])

plt.plot(what_to_plot[:, 1], what_to_plot[:, 0], "-x")
plt.grid()
plt.show()


print("Done")

