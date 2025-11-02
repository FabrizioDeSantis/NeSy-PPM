# NeSyPMM

This repository contains the code for NeSyPPM, a project exploring the integration of Neuro-Symbolic AI for Predictive Process Monitoring.

## Files

*   **`main_bpi12.py`**: Contains the code for evaluating the approach on the BPIC2012 event log.
*   **`main_bpi17.py`**: Contains the code for evaluating the approach on the BPIC2017 event log.
*   **`main_sepsis.py`**: Contains the code for evaluating the approach on the SEPSIS event log.
*   **`main_traffic.py`**: Contains the code for evaluating the approach on the TRAFFIC FINES dataset.
*   **`main_bpi12.py`**: Contains the code for evaluating the approach on the BPIC2012 event log.
*   **`main_bpi17.py`**: Contains the code for evaluating the approach on the BPIC2017 event log.
*   **`main_sepsis.py`**: Contains the code for evaluating the approach on the SEPSIS event log.
*   **`main_traffic.py`**: Contains the code for evaluating the approach on the TRAFFIC FINES dataset.
*   **`data/dataset.py`**: Dataset class.
*   **`metrics.py`**: Contains the code for computing the metrics for the different configurations.
*   **`logs`**: Folder that contains the scripts used to preprocess the event logs.
*   **`model/lstm.py`**: Contains the architecture used for the LSTM/LTN.
*   **`model/transformer.py`**: Contains the architecture used for the Transformer/LTN.
*   **`extract_ltl_rules.py`**: Contains the code for extracting LTL rules using Declarative mining
*   **`knowledge_base.py`**: Contains the logical rules used for training the neuro-symbolic LTN.

The tested event logs can be found at https://data.4tu.nl/articles/dataset