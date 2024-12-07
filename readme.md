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

### View Results

Use the code in the `print_results.ipynb` notebook to view the results.