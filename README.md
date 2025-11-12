# Two-Stage LTN w/ Rule Pruning

This repository contains the code for the paper "Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning"

## Files

*   **`main_bpi12.py`**: Contains the code for the ablation study on the BPIC2012 event log.
*   **`main_bpi17.py`**: Contains the code for the ablation study on the BPIC2017 event log.
*   **`main_sepsis.py`**: Contains the code for the ablation study on the SEPSIS event log.
*   **`main_traffic.py`**: Contains the code for the ablation study on the TRAFFIC FINES dataset.
*   **`data/preprocess_bpi12.py`**: Contains the code for preprocessing the BPIC2012 event log.
*   **`data/preprocess_bpi17.py`**: Contains the code for preprocessing the BPIC2017 event log.
*   **`data/preprocess_sepsis.py`**: Contains the code for preprocessing the Sepsis event log.
*   **`data/preprocess_traffic.py`**: Contains the code for preprocessing the Traffic fines event log.
*   **`model/lstm.py`**: Contains the architecture used for the LSTM backbone.
*   **`model/transformer.py`**: Contains the architecture used for the Transformer backbone.
*   **`data/dataset.py`**: Dataset class.

The tested event logs can be found at https://data.4tu.nl/articles/dataset