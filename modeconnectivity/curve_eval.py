import torch
import torchmetrics

def curve_predict(curve, samplesize=None,  test_loader=None, device="cpu", logger_info=print, eval_straight_line=False, verbose=False, ts=None, model_maker=None, classification_task=True):
    """
    Collect raw model predictions at each sampled point along the curve.
 
    Iterates over a grid of t values in [0, 1], builds the corresponding
    Bézier-interpolated model at each point, and runs it over the full test set.
    Optionally evaluates the straight-line interpolation between w1 and
    w2 instead of the Bézier curve, for comparison.
 
    Args:
        curve (Curve): A fitted Curve object.
        samplesize (int | None): Number of evenly-spaced t values to sample.
            Ignored if ts is provided explicitly.
        test_loader (DataLoader): DataLoader for the evaluation set.
        device (str): Device to run inference on.
        logger_info (callable): Logging function.
        eval_straight_line (bool): If True, evaluate the linear interpolation
            (1-t)*w1 + t*w2 instead of the Bézier curve.
        verbose (bool): Unused; reserved for future verbosity control.
        ts (Tensor | None): Explicit 1-D tensor of t values. If provided,
            samplesize is ignored.
        model_maker (callable | None): Factory for creating blank model instances.
            Defaults to type(curve.model_theta).
        classification_task (bool): If True, true_y is stored as a 1-D integer
            tensor of class indices. If False, as a (N, 1) float tensor.
 
    Returns:
        all_predictions (Tensor): Shape (N, S, C) — raw logits for each
            observation, sample point, and output dimension.
        true_y (Tensor): Shape (N,) or (N, 1) — ground-truth labels.
        ts (Tensor): The t values that were evaluated.
    """
    
    logger_info("")
    logger_info("begin evaluation of curve")
    N_obs = len(test_loader.dataset)
    print(f"{N_obs = }")
    if classification_task:
        N_classes = len(test_loader.dataset.classes)
        #print(f"{N_classes = }")
        true_y = torch.zeros((N_obs), device=device)
    else:
        N_classes = 1
        true_y = torch.zeros((N_obs, 1), device=device)
    if ts is None:
        ts = torch.linspace(0, 1, samplesize, device=device)
    else:
        samplesize = len(ts)
    all_predictions = torch.zeros(N_obs, samplesize, N_classes, device=device)
    
    
    if model_maker is None:
        model_maker = type(curve.model_theta)

    if eval_straight_line:
        w1 = torch.nn.utils.parameters_to_vector(curve.model_start.parameters())
        w2 = torch.nn.utils.parameters_to_vector(curve.model_end.parameters())
        line_model = model_maker().to(device)

    with torch.no_grad():
        for model_no, t in enumerate(ts):
            # set the model parameters to the bezier and linear interpolation
            if eval_straight_line:
                line_model_params = (1-t)*w1.unsqueeze(1) + t*w2.unsqueeze(1)
                torch.nn.utils.vector_to_parameters(line_model_params, line_model.parameters())
                line_model.eval()
            else:
                curve.t=t
                curve.sample_model()
                curve.eval()
                curve.sampled_model.eval()
            print(f"Model {model_no+1} out of {samplesize}", end="\r")

            for i, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                if eval_straight_line:
                    test_pred = line_model(test_x).detach().clone()
                else:
                    test_pred = curve.sampled_model(test_x).detach().clone()
                all_predictions[i * test_loader.batch_size: i * test_loader.batch_size + test_pred.shape[0], model_no,:] = test_pred
                if model_no == 0:
                    true_y[i * test_loader.batch_size: i * test_loader.batch_size + test_y.shape[0]] = test_y
    return all_predictions, true_y, ts

def curve_eval_regression(curve,  test_loader=None, device="cpu", logger_info=print, 
                          eval_straight_line=False, verbose=False, metrics_dict={}, 
                          ts=None, model_maker=None, target_sigma=1.0):
    """
    Evaluate a fitted curve on a regression task.
 
    Samples models along the curve and computes each metric in metrics_dict
    both per sample point and for the ensemble (averaged predictions across
    all points).
 
    Args:
        curve (Curve): A fitted Curve object.
        test_loader (DataLoader): DataLoader for the evaluation set.
        device (str): Device to run inference on.
        logger_info (callable): Logging function.
        eval_straight_line (bool): If True, evaluate linear interpolation
            instead of the Bezier curve (passed through to curve_predict).
        verbose (bool): Passed through to curve_predict.
        metrics_dict (dict[str, callable]): Metrics to compute. Each callable
            should have the signature (pred, target) -> scalar Tensor.
        ts (Tensor | None): Explicit t values to evaluate at. If None,
            a uniform grid of size samplesize is used.
        model_maker (callable | None): Factory for blank model instances.
        target_sigma (float): Reserved for future NLL computation; currently unused.
 
    Returns:
        perpoint_score_dict (dict[str, Tensor]): Per-t metric values,
            each a 1-D Tensor of length S.
        ts (Tensor): The t values that were evaluated.
        ensemble_measurement_dict (dict): Nested dict with keys
            "Start model", "End model", and "Ensemble", each
            mapping metric names to scalar float values.
        all_predictions (Tensor): Raw predictions of shape (N, S, 1).
        true_y (Tensor): Ground-truth targets of shape (N, 1).
    """
    all_predictions, true_y, _ = curve_predict(curve=curve, 
                                               test_loader=test_loader, 
                                               device=device, 
                                               logger_info=logger_info, 
                                               eval_straight_line=eval_straight_line, 
                                               verbose=verbose, ts=ts, 
                                               model_maker=model_maker, classification_task=False)
    samplesize = all_predictions.shape[1]
    ensemble_predictions = all_predictions[:, :, :] 
    start_pred = all_predictions[:, 0, 0]
    end_pred = all_predictions[:, samplesize-1, 0]
    mean_ensemble_pred = ensemble_predictions.mean(dim=1).squeeze()
    #var_ensemble_pred = ensemble_predictions.var(dim=1).squeeze()
    #std_ensemble_pred = ensemble_predictions.std(dim=1).squeeze()
    ensemble_measurement_dict = {"Start model": {}, "End model": {}, "Ensemble": {}}
    for metric_name, metric in metrics_dict.items():
        ensemble_measurement = metric(mean_ensemble_pred[:,None], true_y)
        start_measurement = metric(start_pred[:,None], true_y)
        end_measurement = metric(end_pred[:,None], true_y)
        ensemble_measurement_dict["Start model"][metric_name] = start_measurement.item()
        ensemble_measurement_dict["End model"][metric_name] = end_measurement.item()
        ensemble_measurement_dict["Ensemble"][metric_name] = ensemble_measurement.item()
        logger_info(f"{metric_name} start model: {start_measurement}")
        logger_info(f"{metric_name} end mode: {end_measurement}")
        logger_info(f"{metric_name} ensemble: {ensemble_measurement}")
    #ensemble_measurement_dict["Ensemble"]["NLL"] = torch.distributions.Normal(mean_ensemble_pred, (var_ensemble_pred + target_sigma**2).sqrt()).log_prob(true_y).mean().item()
    #logger_info(f"NLL ensemble: {ensemble_measurement_dict['Ensemble']['NLL']}")

    perpoint_score_dict = {}
    for metric_name, metric in metrics_dict.items():
        perpoint_score_dict[metric_name] = torch.zeros(samplesize, device=device)
        for i in range(samplesize):
            measurement = metric(ensemble_predictions[:,i,:], true_y)
            perpoint_score_dict[metric_name][i] = measurement
    logger_info("finished evaluation of curve")
    logger_info("")
    return perpoint_score_dict, ts, ensemble_measurement_dict, all_predictions, true_y


def curve_eval_classification(curve, test_loader=None, device="cpu", logger_info=print, eval_straight_line=False, verbose=False, metrics_dict={}, ts=None, model_maker=None):
    """
    Evaluate a fitted curve on a classification task.
 
    Samples models along the curve, converts raw logits to probabilities via
    softmax, and computes each metric in metrics_dict both per sample point
    and for the ensemble (averaged softmax probabilities across all points).
 
    Args:
        curve (Curve): A fitted Curve object.
        test_loader (DataLoader): DataLoader for the evaluation set.
        device (str): Device to run inference on.
        logger_info (callable): Logging function.
        eval_straight_line (bool): If True, evaluate linear interpolation
            instead of the Bézier curve (passed through to curve_predict).
        verbose (bool): Passed through to curve_predict.
        metrics_dict (dict[str, callable]): Metrics to compute. Each callable
            should have the signature (pred_probs, target) -> scalar Tensor,
            where pred_probs are softmax probabilities of shape (N, C).
        ts (Tensor | None): Explicit t values to evaluate at. If None,
            a uniform grid of size samplesize is used.
        model_maker (callable | None): Factory for blank model instances.
 
    Returns:
        perpoint_score_dict (dict[str, Tensor]): Per-t metric values,
            each a 1-D Tensor of length S.
        ts (Tensor): The t values that were evaluated.
        ensemble_measurement_dict (dict): Nested dict with keys
            "Start model", "End model", and "Ensemble", each
            mapping metric names to scalar float values.
    """
    all_predictions, true_y, _ = curve_predict(curve=curve, 
                                               test_loader=test_loader, 
                                               device=device, 
                                               logger_info=logger_info, 
                                               eval_straight_line=eval_straight_line, 
                                               verbose=verbose, ts=ts, 
                                               model_maker=model_maker, classification_task=True)
    samplesize = all_predictions.shape[1]
    ensemble_predictions = all_predictions[:, :, :] 
    mean_ensemble_pred_probs = ensemble_predictions.softmax(dim=-1).mean(dim=1)
    start_pred_probs = all_predictions[:, 0, :].softmax(dim=-1)
    end_pred_probs = all_predictions[:, samplesize-1, :].softmax(dim=-1)

    ensemble_measurement_dict = {"Start model": {}, "End model": {}, "Ensemble": {}}

    for metric_name, metric in metrics_dict.items():
        ensemble_measurement = metric(mean_ensemble_pred_probs, true_y)
        start_measurement = metric(start_pred_probs, true_y)
        end_measurement = metric(end_pred_probs, true_y)
        ensemble_measurement_dict["Start model"][metric_name] = start_measurement.item()
        ensemble_measurement_dict["End model"][metric_name] = end_measurement.item()
        ensemble_measurement_dict["Ensemble"][metric_name] = ensemble_measurement.item()

        logger_info(f"{metric_name} start model: {start_measurement}")
        logger_info(f"{metric_name} end mode: {end_measurement}")
        logger_info(f"{metric_name} ensemble: {ensemble_measurement}")

    perpoint_score_dict = {}
    for metric_name, metric in metrics_dict.items():
        perpoint_score_dict[metric_name] = torch.zeros(samplesize, device=device)
        for i in range(samplesize):
            measurement = metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
            perpoint_score_dict[metric_name][i] = measurement

    logger_info("finished evaluation of curve")
    logger_info("")

    return perpoint_score_dict, ts, ensemble_measurement_dict

