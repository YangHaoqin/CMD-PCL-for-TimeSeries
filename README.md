# Time Series Class-Incremental Learning via Confidence-guided Mask Distillation and Prototype-guided Contrastive Learning，AAAI 2026
Class-incremental learning (CIL) has recently gained great attention in the field of time series classification.
Existing methods based on knowledge distillation exhibit impressive ability to preserve prior knowledge and overcome catastrophic forgetting, however, their effectiveness faces a major challenge posed by time series data.
Since temporal data are more susceptible to sensor errors and electronic noise,
the distillation process may be negatively affected by noisy knowledge transfer.
To address this issue, we propose a novel confidence-guided mask distillation (CMD) framework,
to prevent the noisy inheritance during distillation. 
The core of CMD lies in a dynamic masking mechanism guided by prediction confidence, 
capable of allocating higher weights to high-confidence time series and substantially suppressing
the influence of low-confidence ones.
Additionally, different from prior work passing a set of feature prototypes to the classifier simply,
we develop prototype-guided contrastive learning (PCL) to alleviate the classifier bias on new
classes, through extra contrastive constraints to push away the feature distributions of old feature prototypes from those of new classes features.
Extensive experiments on three time-series datasets demonstrate that our method significantly outperforms other replay-free CIL approaches in raising average accuracy, as well as decreasing forgetting rate.


This repository will provide the source code, training scripts, and evaluation tools for our proposed framework once the paper is officially released.
