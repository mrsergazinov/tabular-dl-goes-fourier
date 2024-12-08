## README

# Hyperparameter Optimization and Model Training

This project provides scripts to perform hyperparameter optimization, training, and evaluation of machine learning models using PyTorch and Optuna.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── base_models/
│   ├── mlp.py
│   ├── tabTransformer.py
│   ├── modernNCA.py
├── encoders/
│   ├── numEncoders.py
├── configs/
│   ├── hyperparam.yaml
│   ├── [model]_[dataset].yaml
├── optim_hyperparam.py
├── train_eval.py
└── README.md
```


## Hyperparameter Optimization

Run `optim_hyperparam.py` to perform hyperparameter optimization.
```bash
python optim_hyperparam.py --dataset_name <DATASET_NAME> --model_name <MODEL_NAME> [OPTIONS]
```

This script will:
1. Load data, model, and pre-process the data.
2. Perform hyperparameter optimization using Optuna.
3. Save best hyperparameters to a YAML file, `<MODEL_NAME>_<DATASET_NAME>.yaml`.
4. Evaluate model wiith the best hyperparameters and save results to `results.txt`.

*Note:* teh ranges of hyperparameters for each model are defined in `configs/hyperparam.yaml`.

### All Options

- `--dataset_name`: Dataset to use (`adult`, `california_housing`, `otto_group`, `higgs`).
- `--model_name`: Model to train (`MLP`, `TabTransformer`, `ModernNCA`).
- `--num_encoder`: Numerical encoder (`FourierFeatures`, `BinningFeatures`, `ComboFeatures`, or `None`).
- `--num_encoder_trainable`: Make numerical encoder trainable (`--num_encoder_trainable` or `--no_num_encoder_trainable`).
- `--scaler`: Scaler to use (`SquareScalingFeatures` or `None`).
- `--config_file`: Path to the configuration YAML file.
- `--output_file`: Path to save results.
- `--n_trials`: Number of Optuna trials (default: 10).
- `--n_run`: Number of runs for replication (default: 10).


### Examples

**TabTransformer with raw features on the adult dataset:**
```bash
python optim_hyperparam.py --dataset_name adult --model_name TabTransformer --n_trials 50 --output_file results.txt
```

**TabTransformer with trainable frequency Fourier features (like Gorishnyi) on the adult dataset:**
```bash
python optim_hyperparam.py --dataset_name adult --model_name TabTransformer --n_trials 50 --output_file results.txt --num_encoder FourierFeatures --num_encoder_trainable
```

**MLP with scaled random Fourier features (our method) on the adult dataset:**
```bash
python optim_hyperparam.py --dataset_name adult --model_name TabTransformer --n_trials 50 --output_file results.txt --num_encoder FourierFeatures --no_num_encoder_trainable --scaler SquareScalingFeatures
```

**Note:** currently, all random frequncies for Fourier featurers are sampled from the Gaussian distribution. 

### How to add a new dataset

To add a new data:
1. Modify the `load_data` function in `train_eval.py` to load the new dataset. The dataset should yield a `pd.DataFrame` with the features, `X`, and the `pd.Series` with the target, `y`.
2. Add the task type to the dictionary `TASK_TYPES` in the `train_eval.py` file.
3. Add the dataset to the argument parser in the `optim_hyperparam.py` and `train_eval.py` under `--dataset_name`.

### How to add a new model

To add a new model:
1. Create a new model class in the `base_models` directory. The model should inherit from `torch.nn.Module`.
2. Follow examples of the existing models, e.g. `base_models/MLP.py` to define the relevant initialization (e.g. `d_in, d_out` etc.). The model should also implement `fit` and `evaluate` methods.
3. Add the model to the argument parser in the `optim_hyperparam.py` and `train_eval.py` under `--model_name`.
4. Add the model to the dictionary `MODELS` in the `train_eval.py` file.
5. Add model hyperparameter ranges to the `configs/hyperparam.yaml` file for optimization with Optuna.

### How to add a new numerical encoder

To add a new numerical encoder:
1. Create a new encoder class in the `encoders` directory. The encoder should inherit from `torch.nn.Module`.
2. Follow examples of the existing encoders, e.g. `encoders/numEncoders.py` to define the relevant initialization (e.g. `d_in, d_out` etc.).
3. Add the encoder to the argument parser in the `optim_hyperparam.py` and `train_eval.py` under `--num_encoder`.
4. Add the encoder to the dictionary `NUM_ENCODERS` in the `train_eval.py` file.
5. Add encoder hyperparameter ranges to the `configs/hyperparam.yaml` file for optimization with Optuna.

### View Results

Use the code in the `print_results.ipynb` notebook to view the results.