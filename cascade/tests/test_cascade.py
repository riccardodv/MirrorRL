import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from cascade.nn import CascadeNN, CascadeNNBN




class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            # self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            # output = self.sigmoid(output)
            return output



class CosDataset(TensorDataset):
    def __init__(self, nb_points):
        self.nb_points = nb_points
        self.x = 10 * torch.rand(nb_points, 1) - 5
        self.y = torch.cos(self.x * 3) + 1e-1 * torch.randn(nb_points, 1)
        super().__init__(self.x, self.y)


# def test_reg_inc(nb_points):
#     data = CosDataset(nb_points)
#     idx = torch.arange(len(data.x))
#     train_idx, test_idx = torch.utils.data.random_split(idx, [len(idx) - int(len(idx) * 0.1), int(len(idx) * 0.1)])
#     X_train, Y_train = data.x[train_idx], data.y[train_idx]
#     X_test, Y_test = data.x[test_idx, :], data.y[test_idx, :]
    
#     # train cascade neural network
#     model = CascadeNN(dim_input=1, dim_output=1)
#     loss = nn.MSELoss()

#     min_nsteps = 100000
#     nsteps = 0
#     add_every_n_epoc = 200
#     nb_added_neurones = 20

#     connect_last_k_features = 3 * nb_added_neurones
#     dim_data_input = 1
#     epoch = 0
#     old_weight, old_bias = None, None
#     while nsteps < min_nsteps:
#         if epoch % add_every_n_epoc == 0:
#             # Merging old model with new if any
#             if old_weight is not None:
#                 model.merge_with_old_weight_n_bias(old_weight, old_bias)
#             with torch.no_grad():
#                 train_error = loss(model(X_train), Y_train).item()
#             print(f'nsteps {nsteps} nneurones {model.nb_hidden} training error {train_error}')

#             with torch.no_grad():
#                 test_error = loss(model(X_test), Y_test).item()
#             print(f'nsteps {nsteps} nneurones {model.nb_hidden} test error {test_error}')        
#             print("-------")

#             features = model.get_features(X_train).detach()
#             residuals = Y_train - model(X_train).detach()
#             old_weight, old_bias = model.output.weight.detach().clone(), model.output.bias.detach().clone()
#             data_loader = DataLoader(TensorDataset(features, residuals), batch_size=16, shuffle=True, drop_last=True)  # dataset of features and residuals, the idea from some paper?
            
#             model.add_n_neurones(features, nb_inputs=connect_last_k_features + dim_data_input, n_neurones=nb_added_neurones)  # add just one neuron
#             optim = torch.optim.Adam([*model.cascade_neurone_list[-1].parameters(), *model.output.parameters()], lr=1e-3)
        
#         for feat, residual in data_loader:
#             optim.zero_grad()
#             l2_norm = sum(p.pow(2.0).sum() for p in [*model.cascade_neurone_list[-1].parameters(), *model.output.parameters()])
#             l = loss(model.forward_from_old_cascade_features(feat), residual) + 1e-3 * l2_norm
#             l.backward()
#             optim.step()
#             nsteps += 1
#         epoch += 1

#     model.merge_with_old_weight_n_bias(old_weight, old_bias)

#     # Train feedforward network
#     model2 = Feedforward(1,100)
#     loss = nn.MSELoss()
#     # min_nsteps = 50000
#     nsteps = 0
#     add_every_n_epoc = 200
#     epoch = 0
#     optim = torch.optim.Adam(model2.parameters(), lr=1e-3)
#     while nsteps < min_nsteps:
#         if epoch % add_every_n_epoc == 0:
#             with torch.no_grad():
#                 train_error = loss(model2(X_train), Y_train).item()
#             print(f'nsteps {nsteps} training error {train_error}')

#             with torch.no_grad():
#                 test_error = loss(model2(X_test), Y_test).item()
#             print(f'nsteps {nsteps} test error {test_error}')
#             print("-------")

#             data_loader_train = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True, drop_last=True)
            
#         for x, y in data_loader_train:
#             optim.zero_grad()
#             loss(model2(x), y).backward()
#             optim.step()
#             nsteps += 1
#         epoch += 1

#     # plot results
#     plt.figure()
#     plt.plot(data.x, data.y, 'x')
#     x = torch.linspace(-5, 5, steps=1000)[:, None]
#     with torch.no_grad():
#         y = model(x)
#         y2 = model2(x)
#     plt.plot(x, y, '-', label="cascade")
#     plt.plot(x, y2, '-', label="feedforward")
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
#     test_reg_inc(nb_points=128)



def test_reg_inc(nb_points, min_nsteps):
    data = CosDataset(nb_points)
    idx = torch.arange(len(data.x))
    train_idx, test_idx = torch.utils.data.random_split(idx, [len(idx) - int(len(idx) * 0.1), int(len(idx) * 0.1)])
    X_train, Y_train = data.x[train_idx], data.y[train_idx]
    X_test, Y_test = data.x[test_idx, :], data.y[test_idx, :]
    
    # train cascade neural network
    model = CascadeNN(dim_input=1, dim_output=1)
    loss = nn.MSELoss()

    nsteps = 0
    add_every_n_epoc = 200
    nb_added_neurones = 20

    connect_last_k_features = 3 * nb_added_neurones
    dim_data_input = 1
    epoch = 0
    old_weight, old_bias = None, None
    while nsteps < min_nsteps:
        if epoch % add_every_n_epoc == 0:
            # Merging old model with new if any
            if old_weight is not None:
                model.merge_with_old_weight_n_bias(old_weight, old_bias)
            with torch.no_grad():
                train_error = loss(model(X_train), Y_train).item()
            print(f'nsteps {nsteps} nneurones {model.nb_hidden} training error {train_error}')

            with torch.no_grad():
                test_error = loss(model(X_test), Y_test).item()
            print(f'nsteps {nsteps} nneurones {model.nb_hidden} test error {test_error}')        
            print("-------")

            features = model.get_features(X_train).detach()
            residuals = Y_train - model(X_train).detach()
            old_weight, old_bias = model.output.weight.detach().clone(), model.output.bias.detach().clone()
            data_loader = DataLoader(TensorDataset(features, residuals), batch_size=16, shuffle=True, drop_last=True)  # dataset of features and residuals, the idea from some paper?
            
            model.add_n_neurones(features, nb_inputs=connect_last_k_features + dim_data_input, n_neurones=nb_added_neurones)  # add just one neuron
            optim = torch.optim.Adam([*model.cascade_neurone_list[-1].parameters(), *model.output.parameters()], lr=1e-3)
        
        for feat, residual in data_loader:
            optim.zero_grad()
            l = loss(model.forward_from_old_cascade_features(feat), residual)
            l.backward()
            optim.step()
            nsteps += 1
        epoch += 1

    model.merge_with_old_weight_n_bias(old_weight, old_bias)

    # Train feedforward network
    model2 = Feedforward(1,100)
    loss = nn.MSELoss()
    # min_nsteps = 50000
    nsteps = 0
    add_every_n_epoc = 200
    epoch = 0
    optim = torch.optim.Adam(model2.parameters(), lr=1e-3)
    while nsteps < min_nsteps:
        if epoch % add_every_n_epoc == 0:
            with torch.no_grad():
                train_error = loss(model2(X_train), Y_train).item()
            print(f'nsteps {nsteps} training error {train_error}')

            with torch.no_grad():
                test_error = loss(model2(X_test), Y_test).item()
            print(f'nsteps {nsteps} test error {test_error}')
            print("-------")

            data_loader_train = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True, drop_last=True)
            
        for x, y in data_loader_train:
            optim.zero_grad()
            loss(model2(x), y).backward()
            optim.step()
            nsteps += 1
        epoch += 1

    # plot results
    plt.figure()
    plt.plot(data.x, data.y, 'x')
    x = torch.linspace(-5, 5, steps=1000)[:, None]
    with torch.no_grad():
        y = model(x)
        y2 = model2(x)
    plt.plot(x, y, '-', label="cascade")
    plt.plot(x, y2, '-', label="feedforward")
    plt.legend()
    plt.show()


def test_reg_inc_new(nb_points, min_nsteps):
    data = CosDataset(nb_points)
    idx = torch.arange(len(data.x))
    train_idx, test_idx = torch.utils.data.random_split(idx, [len(idx) - int(len(idx) * 0.1), int(len(idx) * 0.1)])
    X_train, Y_train = data.x[train_idx], data.y[train_idx]
    X_test, Y_test = data.x[test_idx, :], data.y[test_idx, :]

    # train cascade neural network
    model = CascadeNNBN(dim_input=1, dim_output=1)
    loss = nn.MSELoss()

    nsteps = 0
    add_every_n_epoc = 200
    nb_added_neurones = 20

    epoch = 0
    while nsteps < min_nsteps:
        if epoch % add_every_n_epoc == 0:
            # Merging old model with new if any
            model.train(False)
            with torch.no_grad():
                train_error = loss(model(X_train), Y_train).item()
            print(f'nsteps {nsteps} nneurones {model.nb_hidden} training error {train_error}')

            with torch.no_grad():
                test_error = loss(model(X_test), Y_test).item()
            print(f'nsteps {nsteps} nneurones {model.nb_hidden} test error {test_error}')
            print("-------")

            features = model.get_features(X_train).detach()
            data_loader = DataLoader(TensorDataset(features, Y_train), batch_size=16, shuffle=True, drop_last=True)  # dataset of features and residuals, the idea from some paper?

            model.add_n_neurones(nb_added_neurones)  # add just one neuron
            model.train(True)
            optim = torch.optim.Adam(model.parameters(),
                                     lr=1e-1)

        for feat, ty in data_loader:
            optim.zero_grad()
            l = loss(model.forward_from_frozen_features(feat), ty)
            l.backward()
            optim.step()
            nsteps += 1
        epoch += 1

    # plot results
    model.train(False)
    plt.figure()
    plt.plot(data.x, data.y, 'x')
    x = torch.linspace(-5, 5, steps=1000)[:, None]
    with torch.no_grad():
        y = model(x)
    plt.plot(x, y, '-', label="cascade")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    seed = 0

    # torch.manual_seed(seed)
    # test_reg_inc(nb_points=128, min_nsteps=10000)

    torch.manual_seed(seed)
    test_reg_inc_new(nb_points=128, min_nsteps=100000)
