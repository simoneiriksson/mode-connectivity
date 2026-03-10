# Mode-connectivity in loss landscape
This repo is my implementation of the mode-connectivity result in [1](#ref-1). The paper posits that any two local minima in the loss landscape can be connected by a curve through a valley in the loss landscape. Apart from the curiousity of the fact that this is possible, the result can also be used for quick ensemble samling for uncertainty quantification. 
<!--- 
One-paragraph summary: 
- What problem, what you contribute, what the key result is. 
--->

## Key idea
<!--- 
Key idea (theory) in 1–2 screens:
- Definitions/assumptions (minimal).
- Main result(s) as a theorem/claim + intuition.
- A short “why it matters / when it fails” section.
- One figure if you have it.
--->
Assume that we have a class of models $\mathcal M$, parametrized by parameters $w \in \mathbb R^n$. Then the paper posits that two neural networks parametrised with $w_1$ and $w_2$, which both are local minimas of the loss function, can be connected by a simple path, which maintain the same minimal loss.

In the paper, the path is parametrized as a chain of two straight lines that both connect to a third parameter set $\theta$:

$$
\phi_{\theta}(t)=
\begin{cases}
2\left(t\theta + (0.5 - t)w_1\right), & 0 \le t \le 0.5 \\
2\left((t - 0.5)w_2 + (1 - t)\theta\right), & 0.5 \le t \le 1
\end{cases}
$$


Or as I have done in this implementation, the path is parameterized as a bezier curve connecting the start and end points:

$$
\phi_{\theta}(t) = (1-t)^2w_1 + 2t(1-t)\theta + t^2w_2
$$

Given that the two end-point models parametrized by $w_1$ and $w_2$ has been trained and reached local minimas, the curve-model parametrized by $\phi_{\theta}(t)$ is trained by minimizing the expectation over a uniform distribution on the curve:

$$
\ell(\theta) = \underset{t\sim U(0,1)}{\mathbb E}[\mathcal L(\phi_{\theta}(t))]
$$

where $\mathcal L$ is the loss at a single model instance.

The loss is minimized by first sampling $t\sim U(0,1)$ and then generating the model $\phi_{\theta}(t)$ in terms of the tuple $(t, \theta, w_1, w_2)$ by using Pytorch' reparametrization functionality. Then the gradient 
$\nabla_\theta\mathcal L(\phi_{\theta}(t))$ can be calculated and finally a gradient step can be taken.


## Results
The following results are created using the standard setting for  `modeconnectivity.py` as described below. CIFAR10 data has been used for this particular experiment. The model used is a pretty standard convolutional neural network with 3 convolutional layers and two linear layers, ReLU activation and 50% dropout. See [models.py](src/models.py) for further details.

The code first trains the start and end models, and next the $\theta$-model is trained. 
### Loss landscape
The loss landscape, projected unto the plane suspended by the beziercurve is plottet below. Note that the landscape is squeezed such that $w_1$ is mapped to (1,0), $w_2$ is mapped to (0,1) and $\theta$ is mapped to (0,0). 
![Loss landscape plot](experiments/curve_experiment_CIFAR10_CIFAR10ConvNet/figures/loss_landscape.png)

It can indeed be seen that the curve lies in a valley as posited in the paper. 
### Performance metrics along the curve
Given that $\theta$ is now fixed, the performance along the curve can be investigated. In the follwong plot, several performance measures are plotted as function of time $t$. First the Cross Entropy, which is also the loss used for training. Note that the Cross Entropy loss is a slighty bit higher in the inner part of the curve, than at the endpoints. That is reasonable, since the endpoints have been trained freely, whereas the curve is optimized along its full length. However, the difference between the inner parts of the curve and the endpoints is not very large compared to the difference between the endpoints. With regards to the accuracy, the models sampled along the curve are actually *better* than the ones at the endpoints.

![Performance measures along the curve](experiments/curve_experiment_CIFAR10_CIFAR10ConvNet/figures/metric_along_curve.png)
### Ensemble prediction
Finally, the question is: What if use parameter sets sampled along the curve as ensembles? In the normal setup for a classification task, the model would predict logits $\hat{z} = p_{w}(x)$, which would then be turned into probabilites for each class: 
$$
\hat y = \text{softmax}(\hat z).
$$
Instead, in ensemble prediction we use the sampled parameter sets and average over the output: 
$$
\hat z = \frac{1}{K}\sum_i^K p_{\phi_{\theta}(t)}(x)
$$
The idea is that the parameter samples $\phi_{\theta}(t)$ where $t\in U(0,1)$ should estimate the Bayesian posterior distribution roughly. Ok... very rough approximation, but still. 

This should in turn result in a more robust model.
<!--- 
Results:
- Table/plot with the headline outcome.
- Link to results/ artifacts.
--->


## How to run
<!--- 
How to run:
- Installation
- Minimal command to reproduce a quick run
- Full reproduction (optional)
- Hardware/time caveats (brief, factual)
--->
Create a virtual environment and install the packages:
```
python -m pip install -e .
```
Then the code can be run with the standard settings with
```
cd modeconnectivity
python3 modeconnectivity.py
```
## Repo layout
<!--- 
Repo layout
- What’s in src/, evaluation/, etc.
--->

```
mode-connectivity/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── train/
│   └── test/
├── experiments/
│   └── curve_experiment_CIFAR10_CIFAR10ConvNet/
│       ├── curve_model/
│       ├── end_model/
│       ├── figures/
│       ├── logs/
│       ├── models/
│       └── start_model/
├── notebooks/
├── scripts/
└── src/
	├── curve_eval.py
	├── curve_plots.py
	├── modeconnectivity.py
	├── models.py
	├── scheduler.py
	└── train.py
```

- `LICENSE`: License for the project.
- `README.md`: Project overview, results, and usage instructions.
- `requirements.txt`: Python dependencies.
- `data/`: Local training and test datasets (MNIST, FashionMNIST, CIFAR-10).
- `experiments/`: Saved outputs from runs (models, logs, plots, and artifacts).
- `notebooks/`: Interactive notebooks for exploration and analysis.
- `scripts/`: Utility scripts for running or automating experiments.
- `src/`: Core source code for training, curve optimization, evaluation, and plotting.
	- `train.py`: Standard model training routines.
	- `modeconnectivity.py`: Main script to train endpoint models and fit the curve model.
	- `models.py`: Model architectures used in experiments.
	- `scheduler.py`: Learning-rate scheduling logic.
	- `curve_eval.py`: Evaluation utilities for models along the curve.
	- `curve_plots.py`: Plotting utilities for landscapes and curve metrics.

<!--- 
Citation / attribution
- BibTeX or a short citation line if it maps to a paper/report.
--->

## References

1. <a id="ref-1"></a> Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, and Andrew Gordon Wilson. 2018. Loss surfaces, mode connectivity, and fast ensembling of DNNs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 8803–8812. https://arxiv.org/pdf/1802.10026
