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

### Create Conda Environment

1. Create the environment from the file
   ```sh
   conda env create -f environment.yml
   ```

2. Activate the environment `tscl`
   ```sh
   conda activate CMD
   ```
-------------------------------------------------------------------------------------------
## Dataset
### Available Datasets
1. [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. [Dailysports](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) 
3. [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)

### Data Prepareation
We process each dataset individually by executing the corresponding `.py` files located in `data` directory. This process results in the formation of training and test `np.array` data, which are saved as `.pkl` files in `data/saved`. The samples are processed into the shape of (𝐿,𝐶).

### Adding New Dataset
1. Create a new python file in the `data` directory for the new dataset.
2. Format the data into discrete samples in format of numpy array, ensuring each sample maintains the shape of (𝐿,𝐶). Use downsampling or sliding window if needed.
3. If the dataset is not pre-divided into training and test subsets, perform the train-test split manually.
4. Save the numpy arrays of training data, training labels, test data, and test labels into `x_train.pkl`, `state_train.pkl`,`x_test.pkl`, `state_test.pkl` in a new folder in `data/saved`.
5. Finally, add the necessary information of the dataset in `utils/setup_elements.py`.

### Adding New Algorithm
1. Create a new python file in the `agent` directory for the new algorithm.
2. Create a subclass that inherits from the `BaseLearner` class in `agent/base.py`.
3. Customize methods including `train_epoch()`, `after_task()`, `learn_task()` and so on, based on your needs.
4. Add the new algorithm to `agents` in `agents/utils/name_match.py`. If memory buffer is used, add it into `agents_replay` as well.
5. Add the hyperparameters and their ranges for the new algorithm into `config_cl` within `experiment/tune_config.py`.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started


### Setup
1. Download the processed data from [Google Drive](https://drive.google.com/drive/folders/1EFdD07myqmqHhRsjeQ83MdF8gHZXDWLR?usp=share_link). Put it into `data/saved` and unzip
   ```sh
   cd data/saved
   unzip <dataset>.zip
   ```
   You can also download the raw datasets and process the data with the corresponding python files.
2. Revise the following configurations to suit your device:
    * `resources` in `tune_hyper_params` in `experiment/tune_and_exp.py` (See [here](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html) for details)
    * GPU numbers in the `.sh` files in `shell`.

### Run Experiment

There are two functions to run experiments. Set the arguments in the corresponding files or in the command line.

1. Run CIL experiments with custom configurations in `main.config.py`. Note that this function cannot tune/change the hyperparameters for multiple runs. It is recommended for use in sanity checks or debugging.
   ```sh
   python main_config.py
   ```
### Custom Experiment Setup
Change the configurations in 
* `utils/setup_elements.py`: Parameters for data and task stream, including Number of tasks / Number of classes per task / Task split
* `experiment/tune_config.py`: Parameters for `main_tune.py` experiments, such as Memory Budget / Classifier Type / Number of runs / Agent-specific parameters, etc.

For ablation study, revise the corresponding parameters in `experiment/tune_config.py` and rerun the experiments.

For online continual learning, set `epochs` to 1 and `er_mode` to `online`. (beta)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

This repository will provide the source code, training scripts, and evaluation tools for our proposed framework once the paper is officially released.
