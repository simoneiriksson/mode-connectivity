import torch
import torch.nn as nn
import torch.nn.utils.parametrize

class CurveParameterization(nn.Module):
    def __init__(self, curve_fn, t,  param_start, param_end, param_theta):
        super(CurveParameterization, self).__init__()
        self.param_start = param_start
        self.param_end = param_end
        self.param_theta = param_theta
        self.t = t
        self.curve_fn = curve_fn

    def forward(self, x):
        return self.curve_fn(self.param_start, self.param_end, self.param_theta, self.t)

class Curve(nn.Module):
    def __init__(self, model_start, model_end, curve_fn, device="cpu"):
        super(Curve, self).__init__()
        assert type(model_start) == type(model_end)
        self.model_start = model_start.requires_grad_(False)
        self.model_end = model_end.requires_grad_(False)
        self.model_theta = type(model_start)().to(device)
        self.sampled_model = type(model_start)().to(device)
        self.curve_fn = curve_fn
        self.t = 0
        self.initiate_theta()
        self.device=device

    def initiate_theta(self):
        for param1, param2, param3 in zip(self.model_start.parameters(), 
                                          self.model_end.parameters(), 
                                          self.model_theta.parameters()):
                param3.data = param1.data/2 +  param2.data/2

    def sample_t(self):
        self.t = torch.distributions.Uniform(0, 1).sample()

    def sample_model(self, t=None, verbose=False):
        if t == None:
            t = self.t
        param_dicts = []
        self.sampled_model = type(self.model_start)().to(self.device) # overwrite the model, so as to delete previously made reparametrizations
        for module_start, module_end, module_theta, module_sampled in zip(self.model_start.named_modules(), self.model_end.named_modules(), 
                                                                          self.model_theta.named_modules(), self.sampled_model.named_modules()):
            if verbose:
                print(f"\nModule: {module_start[0]}")
            for param_start, param_end, param_theta, param_sampled in zip(module_start[1].named_parameters(recurse=False), 
                                                                          module_end[1].named_parameters(recurse=False), 
                                                                          module_theta[1].named_parameters(recurse=False), 
                                                                          module_sampled[1].named_parameters(recurse=False)):
                if verbose:
                    print(f"Parameter: {param_sampled[0]}")
                param_dicts.append({"module_name": module_sampled[0], 
                                    "module_sampled": module_sampled[1],
                                    "param_name": param_sampled[0],
                                    "param_start": param_start[1], 
                                    "param_end": param_end[1], 
                                    "param_theta": param_theta[1],
                                    "param_name_theta": param_theta[0],
                                    "module_name_theta": module_theta[0]
                                    })
        if verbose:
            print("\n\nParametrizing model")
        for param_dict in param_dicts:
            if verbose:
                print("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"Module_name: {param_dict['module_name']}")
                print(f"Module_sampled: {param_dict['module_sampled']}")
                print(f"Parameter: {param_dict['param_name']}")
                print(f"Theta Parameter name: {param_dict['param_name_theta']}")
                print(f"Theta Model name: {param_dict['module_name_theta']}")

            param_curve = CurveParameterization(self.curve_fn, t, param_dict["param_start"].data, param_dict["param_end"].data, param_dict["param_theta"])
            torch.nn.utils.parametrize.register_parametrization(param_dict["module_sampled"], param_dict["param_name"], param_curve)
            if verbose:
                print("\nafter reparam:")
                print(f"Module_name: {param_dict['module_name']}")
                print(f"Module_sampled: {param_dict['module_sampled']}")
                print(f"Parameter: {param_dict['param_name']}")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    def forward(self, x):
        return self.sampled_model(x)

