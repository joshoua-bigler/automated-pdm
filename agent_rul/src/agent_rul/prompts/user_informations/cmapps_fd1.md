# CMAPPS 
The CMAPPS dataset (Commercial Modular Aero-Propulsion System Simulation) is a widely used benchmark dataset for prognostics and health management (PHM) research, particularly in the context of aircraft engine degradation and remaining useful life (RUL) prediction. It was developed by NASA and is designed to simulate the behavior of a commercial aircraft engine under various operating conditions and fault scenarios.

## Constraints
- Keep healthy fraction around 0.4
- normalization over all engines
- Apply select by correlatoins
- Apply select topk by mutual information
- Apply target scaling with minmax
- Max window size around 50-70
- Max stride around 5-10
- Prefer seq2val architecture e.g. lstm, gru etc.
- Prefer deep learning based models
- Don't use tsfresh or flatten based feature extraction