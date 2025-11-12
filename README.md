# Two-Stage LTN w/ Rule Pruning

This repository contains the code for the paper "Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning"

## Files

*   **`main_bpi12.py`**: Contains the code for the ablation study on the *BPIC2012* event log.
*   **`main_bpi17.py`**: Contains the code for the ablation study on the *BPIC2017* event log.
*   **`main_sepsis.py`**: Contains the code for the ablation study on the *SEPSIS* event log.
*   **`main_traffic.py`**: Contains the code for the ablation study on the *TRAFFIC FINES* dataset.
*   **`data/preprocess_bpi12.py`**: Contains the code for preprocessing the *BPIC2012* event log.
*   **`data/preprocess_bpi17.py`**: Contains the code for preprocessing the *BPIC2017* event log.
*   **`data/preprocess_sepsis.py`**: Contains the code for preprocessing the *Sepsis* event log.
*   **`data/preprocess_traffic.py`**: Contains the code for preprocessing the *TRAFFIC FINES* event log.
*   **`model/lstm.py`**: Contains the architecture used for the LSTM backbone.
*   **`model/transformer.py`**: Contains the architecture used for the Transformer backbone.
*   **`data/dataset.py`**: Dataset class.

## Datasets

The tested event logs can be found at https://data.4tu.nl/articles/dataset

## Usage

Execute the script of interest with following flags:
* --model_type: "lstm" or "transformer"
* --setting: "compliance" or "temporal"
* --seed: random seed used for parameters and splitting

Example for the *SEPSIS* event log:

```python main_sepsis.py --model_type="lstm" --setting="compliance" --seed=42```