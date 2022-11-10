from .cascade_nn import CascadeNN
from cascade.utils import clone_lin_model


class CascadeQ(CascadeNN):
    def __init__(self, dim_input, dim_output, **kwargs):
        super().__init__(dim_input, dim_output, **kwargs)
        self.qfunc = clone_lin_model(self.output)

    def get_q(self, obs, stack=True):
        return self.qfunc(self.get_features(obs, stack))

    def merge_q(self, old_output_model, coef = 0.1):
        self.output.weight.data = self.output.weight.data* coef
        self.output.bias.data = self.output.bias.data * coef
        self.merge_with_old_weight_n_bias(self.qfunc.weight, self.qfunc.bias)
        self.qfunc = clone_lin_model(self.output)

        self.output.weight.data[:, :old_output_model.weight.shape[1]] += old_output_model.weight
        self.output.bias.data += old_output_model.bias

