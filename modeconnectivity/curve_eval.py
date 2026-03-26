import torch
import torchmetrics

def curve_eval_bck(curve, samplesize=20,  test_loader=None, device="cpu", logger_info=None):
    if logger_info == None: logger_info=print
    logger_info("")
    logger_info("begin evaluation of curve")
    K = samplesize
    models = []
    N_obs = len(test_loader.dataset)
    N_classes = len(test_loader.dataset.dataset.classes)
    print(f"{N_obs = }")
    print(f"{N_classes = }")
    ts = torch.cat([torch.rand(samplesize), torch.tensor([0.0]), torch.tensor([1.0])] ) # sample  t's, and concatenate startmodel and endmodel

    all_predictions = torch.zeros(N_obs, K+2, N_classes, device=device)
    true_y = torch.zeros(N_obs, dtype=torch.long, device=device)
    with torch.no_grad():
        for model_no, t in enumerate(ts):
            # set the model parameters to the bezier and linear interpolation
            curve.t=t
            curve.sample_model()
            logger_info(f"Model {model_no+1} out of {samplesize}")
            for i, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                test_pred = curve.sampled_model(test_x).detach().clone()
                all_predictions[i * test_loader.batch_size: (i+1) * test_loader.batch_size, model_no,:] = test_pred
                if model_no ==0:
                    true_y[i * test_loader.batch_size: (i+1) * test_loader.batch_size] = test_y

    ensemble_predictions = all_predictions[:, 0:K, :]
    start_pred = all_predictions[:, K, :].softmax(dim=-1)
    end_pred = all_predictions[:, K+1, :].softmax(dim=-1)
    # dim 0 is the data dimension, dim 1 is the model dimension, dim 2 is the class dimension
    # we want to average over the model dimension

    mean_ensemble_pred = ensemble_predictions.softmax(dim=-1).mean(dim=1)
    var_ensemble_pred = ensemble_predictions.softmax(dim=-1).var(dim=1)
    std_ensemble_pred = ensemble_predictions.softmax(dim=-1).std(dim=1)

    
    CE_loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    NegLL_loss = torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    ECE_metric = torchmetrics.classification.MulticlassCalibrationError(num_classes=N_classes, n_bins=15).to(device)
    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=N_classes).to(device)
    auroc_metric = torchmetrics.classification.AUROC(task="multiclass", num_classes=N_classes).to(device)
    metric_names = ["Cross Entropy", "LogLikelihood", "Expected Calibration Error", "Accuracy", "AUROC"]
    for metric, metric_name in zip([CE_loss, NegLL_loss, ECE_metric, acc_metric, auroc_metric], metric_names):
        ensemble_measurement = metric(mean_ensemble_pred, true_y)
        start_measurement = metric(start_pred, true_y)
        end_measurement = metric(end_pred, true_y)
        if metric_name == "LogLikelihood":
            ensemble_measurement = -ensemble_measurement
            start_measurement = -start_measurement
            end_measurement = -end_measurement
        
        logger_info(f"{metric_name} start model: {start_measurement}")
        logger_info(f"{metric_name} end mode: {end_measurement}")
        logger_info(f"{metric_name} ensemble: {ensemble_measurement}")

    # scores for each model:
    celoss_scores = torch.zeros(K)
    loglik_scores = torch.zeros(K)
    ece_scores = torch.zeros(K)
    accuracy_scores = torch.zeros(K)
    auc_scores = torch.zeros(K)

    for i in range(K):
        celoss_scores[i] = CE_loss(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        loglik_scores[i] = -NegLL_loss(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        ece_scores[i] = ECE_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        accuracy_scores[i] = acc_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        auc_scores[i] = auroc_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)

    logger_info(f"")
    logger_info(f"Model-wise metrics")
    for score, metric_name in zip([celoss_scores, loglik_scores, ece_scores, accuracy_scores, auc_scores], metric_names):
        logger_info(f"")
        logger_info(f"{metric_name = }")
        logger_info(f"{score = }")
        logger_info(f"{score.mean() = }")
    
    logger_info("finished evaluation of curve")
    logger_info("")




def curve_eval(curve, samplesize=20,  test_loader=None, device="cpu", logger_info=print, eval_straight_line=False, verbose=False, metrics_dict={}, ts=None):
    logger_info("")
    logger_info("begin evaluation of curve")
    N_obs = len(test_loader.dataset)
    N_classes = len(test_loader.dataset.dataset.classes)
    print(f"{N_obs = }")
    print(f"{N_classes = }")
    #ts = torch.cat([torch.tensor([0.0]), torch.rand(samplesize-2), torch.tensor([1.0])]).to(device) # sample  t's, and concatenate startmodel and endmodel
    if ts is None:
        ts = torch.linspace(0, 1, samplesize, device=device)
    else:
        samplesize = len(ts)
    all_predictions = torch.zeros(N_obs, samplesize, N_classes, device=device)
    true_y = torch.zeros(N_obs, dtype=torch.long, device=device)
    if eval_straight_line:
        w1 = torch.nn.utils.parameters_to_vector(curve.model_start.parameters())
        w2 = torch.nn.utils.parameters_to_vector(curve.model_end.parameters())
        line_model = type(curve.model_theta)().to(device)

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
                all_predictions[i * test_loader.batch_size: (i+1) * test_loader.batch_size, model_no,:] = test_pred
                #del test_pred  # Free memory
                if model_no == 0:
                    true_y[i * test_loader.batch_size: (i+1) * test_loader.batch_size] = test_y

    ensemble_predictions = all_predictions[:, :, :]
    mean_ensemble_pred_probs = ensemble_predictions.softmax(dim=-1).mean(dim=1)

    start_pred_probs = all_predictions[:, 0, :].softmax(dim=-1)
    end_pred_probs = all_predictions[:, samplesize-1, :].softmax(dim=-1)
    # dim 0 is the data dimension, dim 1 is the model dimension, dim 2 is the class dimension
    # we want to average over the model dimension


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

        # celoss_scores[i] = CE_loss(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        # loglik_scores[i] = -NegLL_loss(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        # ece_scores[i] = ECE_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        # accuracy_scores[i] = acc_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)
        # auc_scores[i] = auroc_metric(ensemble_predictions[:,i,:].softmax(dim=-1), true_y)

    # logger_info(f"")
    # logger_info(f"Model-wise metrics")
    # for score, metric_name in zip([celoss_scores, loglik_scores, ece_scores, accuracy_scores, auc_scores], metric_names):
    #     logger_info(f"")
    #     logger_info(f"{metric_name = }")
    #     logger_info(f"{score = }")
    #     logger_info(f"{score.mean() = }")
    
    logger_info("finished evaluation of curve")
    logger_info("")

    return perpoint_score_dict, ts, ensemble_measurement_dict

