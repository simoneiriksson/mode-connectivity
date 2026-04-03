# %% [markdown]
# In this notebook we train two models (model_start and model_end) and then a third model_theta.

# %%
from models import MyNet, Lenet5, tiny, MyNet_small, CIFAR10ConvNet
import torch
import torch.nn as nn
from scheduler import make_diy_scheduler
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from train import train
from curve_model import Curve
import os
import matplotlib.pyplot as plt
import torchmetrics
from curve_plots import plot_Curve_losslandscape, affine_subspace, bezier_plot
from curve_eval import curve_eval_classification, curve_predict, curve_eval_regression
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

#from getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data, get_dataloader_scipy
from contextlib import contextmanager

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device="cpu"

root = "."
datafolder = f"{root}/data"
base_directory = f"{root}/experiments/results_notebook"
batch_size = 128

# %%

@contextmanager
def torch_seed(seed):
    """
    A context manager to temporarily set the random seed in PyTorch.
    
    Args:
        seed (int): The seed value to use within the context.
    """
    # Save the current random state
    random_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(random_state)


# %%

class FunctionApproximatorModel(torch.nn.Module):
    def __init__(self, num_features=1, hidden_layers=[10], num_outputs=1, nonlin = torch.nn.ReLU(), seed=2):
        super(FunctionApproximatorModel, self).__init__()
        with torch_seed(seed):
            self.layers = [num_features] + hidden_layers + [num_outputs]
            self.hidden_layers = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
            self.nonlin = nonlin
            for layer in self.hidden_layers:
                torch.nn.init.kaiming_normal_(layer.weight)
                #torch.nn.init.constant_(layer.bias, 0.0)
    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.nonlin(layer(x))
        return self.hidden_layers[-1](x)

num_features=1
MODEL = FunctionApproximatorModel
def return_model(MODEL, **model_kargs):
    #model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": seed}
    return lambda: MODEL(**model_kargs)

model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": 47}
UFA_model = return_model(FunctionApproximatorModel, **model_kargs)()
#UFA_model = FunctionApproximatorModel(num_features=num_features, hidden_layers=[10,10], num_outputs=1, nonlin = torch.nn.Tanh(), seed=47)
#return_model(FunctionApproximatorModel, seed=47)
# %%

def make_loaders(X, y, train_size, batch_size=0):
    indices = torch.randperm(len(X))
    #print(f"{X.shape = }")
    #print(f"{y.shape = }")
    X_ = X[indices]
    y_ = y[indices]
    X_train, X_test = X_[:train_size], X_[train_size:]
    y_train, y_test = y_[:train_size], y_[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    subset_test = Subset(test_dataset, indices=range(len(test_dataset) // 1))
    subset_train = Subset(train_dataset, indices=range(len(train_dataset) // 1))
    if batch_size == 0:
        batch_size = max(len(subset_train), len(subset_test))
    train_loader = DataLoader(subset_train, batch_size=batch_size)
    test_loader = DataLoader(subset_test, batch_size=batch_size)
    return train_loader, test_loader
    
# function that takes a model, some observation noise, and generates data
def gen_model_data(model, input_dist, num_train_samples=10, 
                                   num_test_samples=10, noise_std=None, output_dist = None, seed=2, batch_size=0):
    N = num_train_samples + num_test_samples
    if output_dist == None:
        output_dist = lambda x: torch.distributions.Normal(x, noise_std)
    with torch_seed(seed):
        X = input_dist(N)
        modelout = model(X).detach()
        #print(f"{modelout.shape = }")
        #y = modelout + torch.randn_like(modelout) * noise_std
        y = output_dist(modelout).sample()
        #print(f"{y.shape = }")
        #print(f"{X.shape = }")
    return make_loaders(X, y, num_train_samples, batch_size=batch_size)

prior_sigma = 1.
target_sigma = 0.05

start_x = -2.5
end_x = 2.5
#input_dist = lambda x: torch.rand(x, num_features)*(end_x-start_x) + start_x
def input_dist(N):
    xs = torch.rand(N*10, num_features)*(end_x-start_x) + start_x
    index = (1-((xs > -.5) & (xs < 0.5)).int()).nonzero()[:,0]
    return xs[index][:N]

train_loader, test_loader = gen_model_data(UFA_model, input_dist, num_train_samples=500, 
                                           num_test_samples=100, noise_std=target_sigma, seed=2, batch_size=batch_size)

pre_trained_params = torch.nn.utils.parameters_to_vector(UFA_model.parameters())
xs, ys = train_loader.dataset.tensors
xs_plt = torch.linspace(start_x, end_x, 100).unsqueeze(1)
plt.scatter(xs[:, 0].detach(), ys.detach(), c="b", label="data")
ys_plt_pretrained = UFA_model(xs_plt).detach()
plt.plot(xs_plt[:, 0].detach(), ys_plt_pretrained, c="r", label="true function")
plt.show()

dataset = "function"
print(ys_plt_pretrained.shape)


# %%

def NegLogLik_classification():
    def fn(pred, target):
        return torch.nn.CrossEntropyLoss(reduction="mean")(pred, target)
    return fn

def NegLogLik_regression(target_sigma=1.0):
    def fn(pred, target):
        #loss = (pred - target).pow(2).sum()/(2 * target_sigma**2)
        loss = torch.nn.MSELoss(reduction="mean")(pred, target)/(2 * target_sigma**2)
        return loss
    return fn

def loss_func_from_target_sigma(loss_fn, target_sigma):
    if loss_fn is None and target_sigma is not None:  # assume regression
        loss_fn = NegLogLik_regression(target_sigma=target_sigma)

    if loss_fn is None and target_sigma is None:  # assume classification
        loss_fn = NegLogLik_classification()
    return loss_fn

loss_fn = loss_func_from_target_sigma(None, target_sigma)
#num_features=num_features, hidden_layers=[10,10], num_outputs=1, nonlin = torch.nn.Tanh(), seed=47#


# %%
os.makedirs(f"{base_directory}/models", exist_ok=True)
os.makedirs(f"{base_directory}/figures", exist_ok=True)


retrain=False
if retrain:
    model_lr_start = torch.tensor(0.001)
    model_lr_end = torch.tensor(0.001)
    model_epochs = 10000

    total_iter = model_epochs*train_loader.__len__()
    model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": 3}
    model_start = return_model(MODEL, **model_kargs)().to(device)
    optimizer_start = torch.optim.Adam(params=model_start.parameters(), lr=model_lr_start.clone())
    scheduler_start = make_diy_scheduler(optimizer_start, 
                                             train_num_steps=total_iter, 
                                             lr_start_warmup=model_lr_start.clone(), 
                                             lr=model_lr_end.clone(), 
                                             lr_warmup_steps=5*train_loader.__len__(), 
                                             lr_finetune_halftime=total_iter // (5*3), 
                                             lr_finetune_steps=total_iter // 3
            )
    model_start, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(model_start, 
                                                train_loader=train_loader, 
                                                test_loader=test_loader, 
                                                optimizer=optimizer_start, 
                                                scheduler=scheduler_start, 
                                                epochs=model_epochs, loss_fn=loss_fn, device=device, 
                                                plot=True, plotpath=f"{base_directory}/start_model/figures",
                                                verbose=True, print_every_n_epoch=50
                                                )
    torch.save(model_start, f"{base_directory}/models/model_start_{MODEL.__name__}_{dataset}.pth")
    #for k in plots.keys():
    #    display(plots[k][0])
else:
    model_start = torch.load(f"{base_directory}/models/model_start_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)

# %%
plt.scatter(xs[:, 0].detach(), ys.detach(), c="b", label="data")
print(xs_plt.device)
ys_plt_trained = model_start(xs_plt).detach()
plt.plot(xs_plt[:, 0].detach(), ys_plt_trained, c="g", label="post-trained function")
plt.plot(xs_plt[:, 0].detach(), ys_plt_pretrained, label="generating function", linestyle="--", c="r")
plt.legend()
#plt.savefig(f"{fig_folder}/post_trained_function.png")

# %%
if retrain:
    model_lr_start = torch.tensor(0.001)
    model_lr_end = torch.tensor(0.001)
    model_epochs = 10000

    total_iter = model_epochs*train_loader.__len__()

    model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": 2}
    model_end = return_model(MODEL, **model_kargs)().to(device)
    optimizer_end = torch.optim.Adam(params=model_end.parameters(), lr=model_lr_start.clone())
    scheduler_end = make_diy_scheduler(optimizer_end, 
                                             train_num_steps=total_iter, 
                                             lr_start_warmup=model_lr_start.clone(), 
                                             lr=model_lr_end.clone(), 
                                             lr_warmup_steps=5*train_loader.__len__(), 
                                             lr_finetune_halftime=total_iter // (5*3), 
                                             lr_finetune_steps=total_iter // 3
            )
    model_end, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(model_end, 
                                                train_loader=train_loader, 
                                                test_loader=test_loader, 
                                                optimizer=optimizer_end, 
                                                scheduler=scheduler_end, 
                                                epochs=model_epochs, loss_fn=loss_fn, device=device, 
                                                plot=True, plotpath=f"{base_directory}/end_model/figures",
                                                verbose=True, print_every_n_epoch=50
                                                )
    torch.save(model_end, f"{base_directory}/models/model_end_{MODEL.__name__}_{dataset}.pth")
    #for k in plots.keys():
    #    display(plots[k][0])
else:
    model_end = torch.load(f"{base_directory}/models/model_end_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)

# %%
plt.scatter(xs[:, 0].detach(), ys.detach(), c="b", label="data")
print(xs_plt.device)
ys_plt_trained = model_end(xs_plt).detach()
plt.plot(xs_plt[:, 0].detach(), ys_plt_trained, c="g", label="post-trained function")
plt.plot(xs_plt[:, 0].detach(), ys_plt_pretrained, label="generating function", linestyle="--", c="r")
plt.legend()
#plt.savefig(f"{fig_folder}/post_trained_function.png")

# %%
def curve_fn(param_start, param_end, param_theta, t):
    return param_start * (1-t)**2 + param_end * t**2 + param_theta * 2*t*(1-t)

# %%
retrain_curve = False
curve_epochs= 20000
curve_lr_start = torch.tensor(1e-7)
curve_lr_end = torch.tensor(1e-7)
curve_optimizer = "SGD"
total_iter = curve_epochs*train_loader.__len__()


# %%

    
if retrain_curve:
    total_iter = curve_epochs*train_loader.__len__()
    model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": 7}
    curve = Curve(model_start=model_start, model_end=model_end, curve_fn=curve_fn, device=device, model_maker=return_model(MODEL, **model_kargs))
    gamma = ((curve_lr_end.log()-curve_lr_start.log())/(curve_epochs*train_loader.__len__())).exp()

    optimizer = torch.optim.SGD(params=curve.model_theta.parameters(), lr=curve_lr_start, momentum=0.9)

    scheduler = make_diy_scheduler(optimizer, 
                                            train_num_steps=total_iter, 
                                            lr_start_warmup=curve_lr_start.clone(), 
                                            lr=curve_lr_end.clone(), 
                                            lr_warmup_steps=5*train_loader.__len__(), 
                                            lr_finetune_halftime=total_iter // (5*3), 
                                            lr_finetune_steps=total_iter // 3
        )
            

    curve, all_train_losses, lrs, epoch_train_losses, test_losses, epoch_train_accuracy, plots = train(curve, 
                                                                        train_loader=train_loader, 
                                                                        test_loader=test_loader, 
                                                                        optimizer=optimizer, 
                                                                        scheduler=scheduler, 
                                                                        epochs=curve_epochs,
                                                                        loss_fn=loss_fn, 
                                                                        device=device, 
                                                                        plot=True, 
                                                                        plotpath=f"{base_directory}/curve_model/figures", 
                                                                        #plotname=f"curvefitting_{MODEL.__name__}_{dataset}", 
                                                                        modeltype="curve", 
                                                                        verbose=True, print_every_n_epoch=50)
    torch.save(curve.model_theta, f"{base_directory}/models/curve.model_theta_{MODEL.__name__}_{dataset}.pth")
    #for k in plots.keys():
    #    display(plots[k][0])
else:
    curve = Curve(model_start=model_start, model_end=model_end, curve_fn=curve_fn, device=device, model_maker=return_model(MODEL, **model_kargs))
    curve.model_theta = torch.load(f"{base_directory}/models/curve.model_theta_{MODEL.__name__}_{dataset}.pth", map_location=torch.device(device), weights_only=False)
    

# %%
metrics_dict = {
    "loss": loss_fn,
    "MSE": torch.nn.MSELoss(reduction="mean"),
}


# %%
model_kargs = {"num_features": num_features, "hidden_layers": [10,10], "num_outputs": 1, "nonlin": torch.nn.Tanh(), "seed": 4}


fig, ax = plot_Curve_losslandscape(curve, device, f"{base_directory}/figures", train_loader, N_points=21, loss_fn=loss_fn, recalc_mesh=True, N_bezierpoints=30, model_maker=return_model(MODEL, **model_kargs))
fig.savefig(f"{base_directory}/figures/loss_landscape.png")
#plt.show(fig)
samplesize = 30
# all_predictions, true_y, ts = curve_predict(curve, samplesize=samplesize, test_loader=test_loader, 
#                                    device=device, logger_info=print, 
#                                    eval_straight_line=False, verbose=False, 
#                                    model_maker=return_model(MODEL, **model_kargs), 
#                                    classification_task=False)
# ensemble_predictions = all_predictions[:, :, :] 
# start_pred = all_predictions[:, 0, 0]
# end_pred = all_predictions[:, samplesize-1, 0]
# mean_ensemble_pred = ensemble_predictions.mean(dim=1).squeeze()
# var_ensemble_pred = ensemble_predictions.var(dim=1).squeeze()
# std_ensemble_pred = ensemble_predictions.std(dim=1).squeeze()
# ensemble_measurement_dict = {"Start model": {}, "End model": {}, "Ensemble": {}}
# for metric_name, metric in metrics_dict.items():
#     ensemble_measurement = metric(mean_ensemble_pred, true_y)
#     start_measurement = metric(start_pred, true_y)
#     end_measurement = metric(end_pred, true_y)
#     ensemble_measurement_dict["Start model"][metric_name] = start_measurement.item()
#     ensemble_measurement_dict["End model"][metric_name] = end_measurement.item()
#     ensemble_measurement_dict["Ensemble"][metric_name] = ensemble_measurement.item()
#     print(f"{metric_name} start model: {start_measurement}")
#     print(f"{metric_name} end mode: {end_measurement}")
#     print(f"{metric_name} ensemble: {ensemble_measurement}")

# perpoint_score_dict = {}
# for metric_name, metric in metrics_dict.items():
#     perpoint_score_dict[metric_name] = torch.zeros(samplesize, device=device)
#     for i in range(samplesize):
#         measurement = metric(ensemble_predictions[:,i,:], true_y)
#         perpoint_score_dict[metric_name][i] = measurement

# eval_results = {"curve_perpoint_score_dict": perpoint_score_dict, "ts": ts, "curve_ensemble_score_dict": ensemble_measurement_dict}

# fig, axs, eval_results = bezier_plot(curve, device, test_loader=train_loader, 
#                                      plottype="linear", eval_results=eval_results,
#                     N_bezierpoints = 30,
#                     plot_linear=False, metrics_dict=metrics_dict, classification_task=False, model_maker=return_model(MODEL, **model_kargs))

# fig.savefig(f"{base_directory}/figures/metric_along_curve.png")
# #plt.show(fig)
# print(pd.DataFrame(eval_results["curve_ensemble_score_dict"]).T.reset_index().rename(columns={"index": "Model"}).to_markdown(index=False))


perpoint_score_dict, ts, ensemble_measurement_dict, all_predictions, true_y = curve_eval_regression(
    curve, samplesize=samplesize,  test_loader=test_loader, device=device, 
    logger_info=print, eval_straight_line=False, verbose=False, metrics_dict=metrics_dict, 
    model_maker=return_model(MODEL, **model_kargs))

eval_results = {"curve_perpoint_score_dict": perpoint_score_dict, "ts": ts, "curve_ensemble_score_dict": ensemble_measurement_dict}
fig, axs, eval_results = bezier_plot(curve, device, test_loader=train_loader, 
                                     plottype="linear", eval_results=eval_results,
                        N_bezierpoints = 30,
                        plot_linear=False, metrics_dict=metrics_dict, 
                        classification_task=False, 
                        model_maker=return_model(MODEL, **model_kargs))

fig.savefig(f"{base_directory}/figures/metric_along_curve_2.png")
#plt.show(fig)
print(pd.DataFrame(eval_results["curve_ensemble_score_dict"]).T.reset_index().rename(columns={"index": "Model"}).to_markdown(index=False))

