import torch
import torch.nn as nn
import torch.nn.utils.parametrize

class CurveParameterization(nn.Module):
    """
    A module that defines the curve parameterization for a single parameter, 
    which can be used to reparametrize the parameters of the sampled model. 
    """
    def __init__(self, curve_fn, t,  param_start, param_end, param_theta):
        
        super(CurveParameterization, self).__init__()
        self.param_start = param_start
        self.param_end = param_end
        self.param_theta = param_theta
        self.t = t
        self.curve_fn = curve_fn

    def forward(self, x):
        """
        Evaluate the Bezier curve at self.t for a single parameter tensor.

        This method is called automatically by PyTorch's parametrize machinery
        during each forward pass of the sampled model. The input x is the
        current raw parameter value (unused — the curve ignores it and computes
        directly from the stored start, end, and theta tensors).

        Returns:
            Tensor: The interpolated parameter value phi(t) as defined by
                curve_fn(param_start, param_end, param_theta, t).
        """
        return self.curve_fn(self.param_start, self.param_end, self.param_theta, self.t)

class Curve(nn.Module):
    def __init__(self, model_start, model_end, curve_fn, device="cpu", model_maker=None, logger_info=print):
        """
        Initialise a Bézier curve connecting two trained models in parameter space.

        Stores the start and end models as frozen references, creates a trainable
        midpoint model (model_theta) initialised to their average, and builds a
        sampled_model that will be reparametrised on each forward pass via curve_fn.

        Args:
            model_start (nn.Module): The first trained endpoint model (w1). Frozen.
            model_end (nn.Module): The second trained endpoint model (w2). Frozen.
            curve_fn (callable): Interpolation function with signature
                (param_start, param_end, param_theta, t) -> Tensor.
            device (str): Device to place models on, e.g. "cpu" or "cuda".
            model_maker (callable | None): Factory that returns a fresh model instance.
                Defaults to type(model_start).
            logger_info (callable): Logging function. Defaults to print.
        """
        super(Curve, self).__init__()
        assert type(model_start) is type(model_end)
        self.model_start = model_start.requires_grad_(False)
        self.model_end = model_end.requires_grad_(False)
        self.model_maker = model_maker
        if self.model_maker is None:
            self.model_maker = type(model_start)
        self.model_theta = self.model_maker().to(device)
        self.sampled_model = self.model_maker().to(device)
        self.curve_fn = curve_fn
        self.t = 0
        self.initiate_theta()
        self.device=device
        self.logger_info = logger_info

    def initiate_theta(self):
        """
        Initialise theta as the midpoint between the start and end models.

        Sets each parameter of model_theta to the arithmetic mean of the
        corresponding parameters in model_start and model_end. This places
        theta on the straight line between w1 and w2, which is a reasonable
        starting point before curve training begins.
        """
        for param1, param2, param3 in zip(self.model_start.parameters(), 
                                          self.model_end.parameters(), 
                                          self.model_theta.parameters()):
                param3.data = param1.data/2 +  param2.data/2

    def sample_t(self):
        self.t = torch.distributions.Uniform(0, 1).sample()

    def sample_model(self, t=None, verbose=False):
        """
        Build a sampled model at position t along the Bezier curve.

        For each parameter in the network, registers a CurveParameterization
        as a PyTorch parametrization (torch.nn.utils.parametrize). This means
        that when the sampled model performs a forward pass, each parameter is
        computed on-the-fly as the Bezier interpolation:

            phi(t) = (1-t)^2 * w1 + 2t(1-t) * theta + t^2 * w2

        where w1, w2 are the fixed start/end parameters and theta is the
        trainable midpoint. Gradients flow through theta only — w1 and w2
        are frozen.

        The sampled_model is recreated from scratch on each call to discard
        any parametrizations registered in a previous call.

        Args:
            t (float | Tensor | None): Position along the curve in [0, 1].
                Defaults to self.t, which is set by sample_t().
            verbose (bool): If True, logs each module and parameter as it is
                parametrized. Useful for debugging new model architectures.
        """
        if t is None:
            t = self.t
        param_dicts = []
        self.sampled_model = self.model_maker().to(self.device) # overwrite the model, so as to delete previously made reparametrizations
        for module_start, module_end, module_theta, module_sampled in zip(self.model_start.named_modules(), self.model_end.named_modules(), 
                                                                          self.model_theta.named_modules(), self.sampled_model.named_modules()):
            if verbose:
                self.logger_info(f"\nModule: {module_start[0]}")
            for param_start, param_end, param_theta, param_sampled in zip(module_start[1].named_parameters(recurse=False), 
                                                                          module_end[1].named_parameters(recurse=False), 
                                                                          module_theta[1].named_parameters(recurse=False), 
                                                                          module_sampled[1].named_parameters(recurse=False)):
                if verbose:
                    self.logger_info(f"Parameter: {param_sampled[0]}")
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
            self.logger_info("\n\nParametrizing model")
        for param_dict in param_dicts:
            if verbose:
                self.logger_info("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                self.logger_info(f"Module_name: {param_dict['module_name']}")
                self.logger_info(f"Module_sampled: {param_dict['module_sampled']}")
                self.logger_info(f"Parameter: {param_dict['param_name']}")
                self.logger_info(f"Theta Parameter name: {param_dict['param_name_theta']}")
                self.logger_info(f"Theta Model name: {param_dict['module_name_theta']}")
                self.logger_info("\n\nMORE INFO ")
                self.logger_info(f"param_start: {param_dict['param_start'].shape}")
                self.logger_info(f"param_end: {param_dict['param_end'].shape}")
                self.logger_info(f"param_theta: {param_dict['param_theta'].shape}")

                

            param_curve = CurveParameterization(self.curve_fn, t, param_dict["param_start"].data, param_dict["param_end"].data, param_dict["param_theta"])
            if verbose:
                self.logger_info(f"param_curve: {param_curve}")
                
            torch.nn.utils.parametrize.register_parametrization(param_dict["module_sampled"], param_dict["param_name"], param_curve)
            if verbose:
                self.logger_info("\nafter reparam:")
                self.logger_info(f"Module_name: {param_dict['module_name']}")
                self.logger_info(f"Module_sampled: {param_dict['module_sampled']}")
                self.logger_info(f"Parameter: {param_dict['param_name']}")
                self.logger_info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    def forward(self, x):
        return self.sampled_model(x)

