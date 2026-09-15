# Two-Stage LTN w/ Rule Pruning

This repository contains the code for the paper
> **Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning**

---

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

---

## Datasets

The event logs used in the study can be downloaded from the following links:

* [BPIC2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)
* [BPIC2017](https://data.4tu.nl/datasets/34c3f44b-3101-4ea9-8281-e38905c68b8d/1)
* [Sepsis](https://data.4tu.nl/datasets/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/1)
* [Traffic fines](https://data.4tu.nl/datasets/806acd1a-2bf2-4e39-be21-69b8cad10909/1)

---

## Translation of Declarative constraints into First-Order Logic formulas

The file **`declare_to_fol_templates.pdf`** contains the translation of declarative constraints intoto first-order logic formulas. The resulting FOL formulas can be implemented in the LTN framework to express control-flow constraints in business processes.

The file **`declare_ltn_templates.py`** implements the declarative constraints described in the previous PDF as predicates that can be used within the LTN framework. **(in progress)**

---

## Reproducibility

Execute the script of interest with following flags:
* --model_type: "lstm" or "transformer"
* --setting: "compliance" or "temporal"
* --seed: random seed used for parameters and splitting
* --num_epochs: number of training epochs of vanilla models
* --num_epochs_nesy: number of training epochs of LTN models
* --hidden_size: hidden_size of LSTM/Transformer backbones
* --num_layers: LSTM/Transformer layers
* --dropout_rate: dropout_rate for LSTM/Transformer backbones

Example for the *SEPSIS* event log with default parameters:

```python main_sepsis.py --model_type="lstm" --setting="compliance" --seed=42```