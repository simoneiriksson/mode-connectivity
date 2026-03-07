import torch
import torch.nn as nn

class IrisNet(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    

class logistic_regression_model(nn.Module):
    def __init__(self,numclasses=2, numfeatures=2, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(logistic_regression_model, self).__init__()
        #self.fc1 = nn.Linear(4, 4)
        self.fc = nn.Linear(numclasses, numfeatures)
        #self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x

class Lenet5(nn.Module):
    # Implemented based on https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
    def __init__(self, seed=None, num_classes=10):
        if seed is not None:
            torch.manual_seed(seed)
        super(Lenet5, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(256, 120)  # 16*4*4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = nn.Flatten(1)(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class tiny(nn.Module):
    def __init__(self, seed=None, num_classes=10):
        if seed is not None:
            torch.manual_seed(seed)
        super(tiny, self).__init__()
        self.fc1 = nn.Linear(28*28, num_classes, bias=False)
        
    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = self.fc1(x)
        return x
    
class small(nn.Module):
    # Implemented based on https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
    def __init__(self, seed=None, num_classes=10):
        super(small, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.fc_final = nn.Linear(6*7*7, num_classes, bias=False) # 6*7*7
        self.nonlin = nn.ReLU()
        self.verbose = False
    def forward(self, x):
        if self.verbose: print(f"1: {x.shape=}")
        x,_ = self.pool(self.nonlin(self.conv1(x)))
        if self.verbose: print(f"2: {x.shape=}")
        x,_ = self.pool(self.nonlin(self.conv2(x)))
        if self.verbose: print(f"3: {x.shape=}")
        x = nn.Flatten(1)(x)
        if self.verbose: print(f"4: {x.shape=}")
        x = self.fc_final(x)
        if self.verbose: print(f"5: {x.shape=}")
        return x



class small2(nn.Module):
    # Implemented based on https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
    def __init__(self, seed=None, num_classes=10, dropout=0.1):
        super(small2, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1) # parameters = 6*(1*3*3+1) = 60, input size = 28*28, output size = 6*28*28 = 4704 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1) # parameters = 12*(6*3*3+1) = 660, input size = 6*14*14, output size = 12*14*14 = 2352
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1) # parameters = 12*(12*3*3+1) = 1308, input size = 12*7*7, output size = 12*7*7 = 588
        self.pool = nn.MaxPool2d(2, return_indices=True)
        num_hideen = 12*3 
        self.lin1 = nn.Linear(3*3*12, num_hideen, bias=False) # paramters = 3*3*12*12*2 = 2592 
        self.lin2 = nn.Linear(num_hideen, num_classes, bias=False) # paramters = 2*12*10 = 240
        self.nonlin = nn.ReLU()
        self.verbose = False
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x,_ = self.pool(self.nonlin(self.conv1(x)))
        x,_ = self.pool(self.nonlin(self.conv2(x)))
        x,_ = self.pool(self.nonlin(self.conv3(x)))
        x = nn.Flatten(1)(x)
        x = self.nonlin(self.lin1(x))
        x = self.lin2(x)
        return x



class mini(nn.Module):
    # Implemented based on https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
    def __init__(self, seed=None, num_classes=10):
        super(mini, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(28*28, num_classes, bias=False)
        self.fc2 = nn.Linear(num_classes, num_classes, bias=False)
        self.nonlin = nn.ReLU()
    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x




class MyNet_unfolded(nn.Module):
    def __init__(self, seed=None, dropout=0.5, num_classes=10):
        super(MyNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.linear1 = nn.Linear(128*7*7, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_classes)
        self.nonlin = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.nonlin(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.nonlin(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batchnorm1(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


class MyNet_small(nn.Module):
    def __init__(self, seed=None, dropout=0.5, num_classes=10):
        super(MyNet_small, self).__init__()
        self.dropout = dropout
        if seed is not None:
            torch.manual_seed(seed)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(32),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.linear_block = nn.Sequential(
            nn.Linear(64*7*7, 64),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x


class MyNet(nn.Module):
    def __init__(self, seed=None, dropout=0.5, num_classes=10):
        super(MyNet, self).__init__()
        self.dropout = dropout
        if seed is not None:
            torch.manual_seed(seed)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.linear_block = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128*7*7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x


class CIFAR10ConvNet(nn.Module):
    def __init__(self, seed=None, dropout=0.3, num_classes=10):
        super(CIFAR10ConvNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class functionestimator(nn.Module):
    def __init__(self):
        super(functionestimator, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.nonlin = nn.Softplus()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        x = self.fc3(x)
        return x


class functionestimator_poly(nn.Module):
    def __init__(self, degree=2):
        super(functionestimator_poly, self).__init__()
        self.num_params = degree
        self.fc1 = nn.Linear(self.num_params, 1, bias=False)
        torch.nn.init.constant_(self.fc1.weight, torch.tensor(0.))
        self.register_buffer("paws", torch.tensor(range(0, self.num_params)))
        fac = torch.tensor([self.paws[1:i+1].prod() for i in range(self.num_params)])
        self.register_buffer("fac", fac)
    def forward(self, x):
        #x = self.batchnorm(x)
        x = (x**self.paws)/(self.fac)
        x = self.fc1(x)
        return x

class tiny_functionestimator(nn.Module):
    def __init__(self):
        super(tiny_functionestimator, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class linear_functionestimator(nn.Module):
    def __init__(self):
        super(linear_functionestimator, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


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





