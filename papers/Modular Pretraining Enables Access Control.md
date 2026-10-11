Blog: https://ae.studio/research/modular-pretraining-access-control
Paper: https://arxiv.org/abs/2607.08077

Threat: the misuse of helpful knowledge for harmful tasks.
Approach: Access control at the level of individual capabilities. The most direct route to achieve this is to train separate models on separately filtered datasets, but this is expensive. To address this, the researchers developed Gradient-Routed Auxiliary Modules (GRAM), a method that approximates the performance of multiple data filtere models at the cost of a single training run.