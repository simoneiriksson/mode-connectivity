import torchmetrics

import torch
import torch.nn as nn
from models import Curve, CurveParameterization
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm

def affine_subspace(curve: Curve) -> tuple:
    # Create affine mapping that spans the subspace
    b = torch.nn.utils.parameters_to_vector(curve.model_theta.parameters())
    w1 = torch.nn.utils.parameters_to_vector(curve.model_start.parameters())
    w2 = torch.nn.utils.parameters_to_vector(curve.model_end.parameters())
    A = torch.column_stack([w1-b, w2-b])
    return A, b

def CurveLossmesh(curve, N_points = 11, x_min = -.2, x_max = 1.2, test_loader=None, loss_fn=None, device="cpu", logger_info=None, verbose=False):
    if logger_info == None: logger_info=print
    logger_info("begin calculation of mesh for loss landscape plot")
    #x_min = -.2; x_max = 1.2;N_points = 11
    A, b = affine_subspace(curve) # get affine subspace matrix and bias vector
    x1s = torch.linspace(x_min, x_max, N_points, device=device)
    y1s = torch.linspace(x_min, x_max, N_points, device=device)

    xs, ys = torch.meshgrid(x1s, y1s, indexing='xy')
    xs.reshape(-1).shape
    mesh = torch.row_stack([xs.reshape(-1), ys.reshape(-1)])

    model_mesh = type(curve.model_theta)().to(device)  # creates a new empty model
    num_obs = N_points**2
    loss_values = torch.zeros(num_obs)

    for mesh_point_no in range(num_obs):
        mesh_point = mesh[:,mesh_point_no]
        input = A@mesh_point + b # 

        torch.nn.utils.vector_to_parameters(input, model_mesh.parameters())
        model_mesh.eval()
        test_loss = 0
        total_obs= 0 
        for i, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            total_obs += len(test_x)
            test_pred = model_mesh(test_x)
            loss = loss_fn(test_pred, test_y)
            test_loss += loss.item() * len(test_x) 
        test_loss /= total_obs
        if verbose: logger_info(f"Mesh-point {mesh_point_no+1} out of {num_obs}: {test_loss = }")
        else: print(f"Mesh-point {mesh_point_no+1} out of {num_obs}: {test_loss = }", end="\r")
        loss_values[mesh_point_no] = test_loss
    return loss_values.reshape(N_points, N_points), xs, ys

def plot_Curve_losslandscape(curve, device, folder, test_loader, N_points=30, loss_fn=None, recalc_mesh=True, logger_info=None, N_bezierpoints=100):
    if logger_info == None: logger_info=print
    logger_info("")
    logger_info("begin loss landscape plot")
    dataset_name = type(test_loader.dataset.dataset).__name__
    model_name = type(curve.model_theta).__name__

    if recalc_mesh:
        loss_values_mesh, xs, ys = CurveLossmesh(curve, N_points=N_points, x_min = -0.5, x_max = 1.5, test_loader=test_loader, 
                                            loss_fn=loss_fn, device=device, logger_info=logger_info, verbose=False)
        torch.save((loss_values_mesh, xs, ys), f"{folder}/lossmesh_{type(curve.model_theta).__name__}_{dataset_name}.pth")
    else:
        loss_values_mesh, xs, ys = torch.load(f"{folder}/lossmesh_{type(curve.model_theta).__name__}_{dataset_name}.pth")
    
    fig, ax = plt.subplots(1, 1)
    # plot contourplot

    #cs = ax.contourf(xs.to("cpu"), ys.to("cpu"), loss_values_mesh.to("cpu"), locator=ticker.LogLocator(subs="auto"))
    #cs = ax.contourf(xs.to("cpu"), ys.to("cpu"), loss_values_mesh.to("cpu"), norm=LogNorm(), levels=10000, cmap='viridis')
    levels = torch.linspace(loss_values_mesh.min().log(), loss_values_mesh.max().log(), 20).exp()
    # Use a log-scaled colormap to match exponential levels
    norm = LogNorm(vmin=levels[0].item(), vmax=levels[-1].item())
    cs = ax.contour(xs.to("cpu"), ys.to("cpu"), loss_values_mesh.to("cpu"), levels=levels, colors="black", linewidths=0.5)
    cf = ax.contourf(xs.to("cpu"), ys.to("cpu"), loss_values_mesh.to("cpu"), levels=levels, norm=norm, cmap=cm.viridis)
    # Colorbar with one tick per contour level
    level_values = levels.detach().cpu().tolist()
    cbar = fig.colorbar(cf, ax=ax, label='Test Loss', ticks=level_values)
    # Fix ticks/labels to exactly match the contour levels
    cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(level_values))
    cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter([f"{v:.3g}" for v in level_values]))
    cbar.minorticks_off()

    x1 = torch.tensor([1., 0.])
    x2 = torch.tensor([0., 1.])
    theta = torch.tensor([0., 0.])

    
    ts = torch.linspace(0, 1, N_bezierpoints)
    inputs = curve.curve_fn(x1.unsqueeze(1), x2.unsqueeze(1), theta.unsqueeze(1), ts)

    ax.plot(inputs[0,:], inputs[1,:], marker="x")
    ax.scatter(x2[0],x2[1], label="model2")
    ax.scatter(x1[0],x1[1], label="model1")
    ax.scatter(0,0 , label="theta")
    ax.legend()
    fig.savefig(f"{folder}/losslandscape_{model_name}_{dataset_name}.png")
    #plt.show()
    plt.close()
    logger_info("finished loss landscape plot")
    logger_info("")


def bezier_plot(curve, device, folder, test_loader, plottype="semilog", N_bezierpoints = 20, loss_fn=None, logger_info=None, verbose=False, plot_linear=True):
    if logger_info == None: logger_info=print
    logger_info("")
    logger_info("begin bezierplot")
    dataset_name = type(test_loader.dataset.dataset).__name__
    model_name = type(curve.model_theta).__name__

    # The affine subspace is made such that the endpoints are mapped to [1,0] and [0,1] and the theta point is mapped to [0,0]
    #x1 = torch.tensor([1., 0.])
    #x2 = torch.tensor([0., 1.])
    #theta = torch.tensor([0., 0.])
    
    # the t's are the time steps we walk along the curve
    ts = torch.linspace(0, 1, N_bezierpoints)
    # this is the points along the curve
    #inputs = curve.curve_fn(x1.unsqueeze(1), x2.unsqueeze(1), theta.unsqueeze(1), ts)
    #A, b = affine_subspace(curve)
    # and these are the corresponding model parameters
    #bezier_models = A@inputs.to(device) + b.unsqueeze(1) 

    # Similarly, we can make a linear interpolation between the two endpoint-models
    w1 = torch.nn.utils.parameters_to_vector(curve.model_start.parameters())
    w2 = torch.nn.utils.parameters_to_vector(curve.model_end.parameters())
    line_models = (1-ts.unsqueeze(0).to(device))*w1.unsqueeze(1) + ts.unsqueeze(0).to(device)*w2.unsqueeze(1)

    # initialize two models
    #model_bezier = type(curve.model_theta)().to(device)
    model_line = type(curve.model_theta)().to(device)

    # initialize tensors for the predictions and true values
    bezier_predictions = torch.zeros(len(test_loader.dataset), N_bezierpoints, len(test_loader.dataset.dataset.classes), device=device)
    line_predictions = torch.zeros(len(test_loader.dataset), N_bezierpoints, len(test_loader.dataset.dataset.classes), device=device)
    true_y = torch.zeros(len(test_loader.dataset), dtype=torch.long, device=device)

    with torch.no_grad():
        #for point_no in range(bezier_models.shape[1]):
        total_obs_test = len(test_loader.dataset)
        for point_no in range(N_bezierpoints):
            # set the model parameters to the bezier and linear interpolation
            curve.t=ts[point_no]
            curve.sample_model()
            torch.nn.utils.vector_to_parameters(line_models[:,point_no], model_line.parameters())
            model_line.eval()
            curve.sampled_model.eval()
            if verbose: logger_info(f"Point {point_no+1} out of {N_bezierpoints}")
            else: print(f"Point {point_no+1} out of {N_bezierpoints}", end="\r")
            for i, (test_x, test_y) in enumerate(test_loader):
                this_batch_size = test_x.shape[0]
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                bezier_pred = curve.sampled_model(test_x)
                bezier_predictions[i * test_loader.batch_size: i * test_loader.batch_size + this_batch_size, point_no ,:] = bezier_pred
                if plot_linear: 
                    line_pred = model_line(test_x)
                    line_predictions[i * test_loader.batch_size: i * test_loader.batch_size + this_batch_size, point_no ,:] = line_pred
                if point_no ==0:
                    true_y[i * test_loader.batch_size: i * test_loader.batch_size + this_batch_size] = test_y

    # initialize a bunch of metrics:
    loss_fn_loss = lambda pred, y: loss_fn(pred, y).item()
    CE_loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    NegLL_loss = torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    #NegLL_loss = torchmetrics.classification.MulticlassNLLLoss(num_classes=10)
    ECE_metric = torchmetrics.classification.MulticlassCalibrationError(num_classes=10, n_bins=15)
    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
    auroc_metric = torchmetrics.classification.AUROC(task="multiclass", num_classes=10)

    # loop over metrics and create plots.
    for metric, metric_name in zip([loss_fn_loss, CE_loss, NegLL_loss, ECE_metric, acc_metric, auroc_metric], 
                        ["loss_fn_loss", "Cross Entropy", "LogLikelihood", "Expected Calibration Error", "Accuracy", "AUROC"]):
        measurement_bezier = []
        measurement_line = []
        for point_no in range(N_bezierpoints):
            if plot_linear: 
                measurement_line += [metric(line_predictions[:,point_no,:].to("cpu"), true_y.to("cpu"))]
            measurement_bezier += [metric(bezier_predictions[:,point_no,:].to("cpu"), true_y.to("cpu"))]
        if metric_name == "LogLikelihood":
            measurement_bezier = -torch.tensor(measurement_bezier)
            if plot_linear:
                measurement_line = -torch.tensor(measurement_line)
        if plottype == "semilog":
            plt.semilogy(ts, measurement_bezier, label="bezier")
            if plot_linear: plt.semilogy(ts, measurement_line, label="linear")
        elif plottype == "linear":
            plt.plot(ts, measurement_bezier, label="bezier")
            if plot_linear: plt.plot(ts, measurement_line, label="linear")
        plt.legend()
        plt.title(f"{metric_name} along curve")
        #plt.show()
        #plt.close()
        logger_info(f"{metric_name} bezier: {measurement_bezier}")
        if plot_linear: logger_info(f"{metric_name} line: {measurement_line}")
        plt.savefig(f"{folder}/metric_along_curve_{plottype}_{metric_name}_{model_name}_{dataset_name}.png")
        plt.close()

    #plt.show()
    plt.close()
    logger_info("finished bezierplot")

