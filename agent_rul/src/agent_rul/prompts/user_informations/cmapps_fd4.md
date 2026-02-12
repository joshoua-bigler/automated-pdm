# CMAPPS 
The CMAPPS dataset (Commercial Modular Aero-Propulsion System Simulation) is a widely used benchmark dataset for prognostics and health management (PHM) research, particularly in the context of aircraft engine degradation and remaining useful life (RUL) prediction. It was developed by NASA and is designed to simulate the behavior of a commercial aircraft engine under various operating conditions and fault scenarios.

## Preprocessing Recommendations
- normalization with per operational setting using opc_mode (kmax=6)
- rul clip at 125
- apply mutual information based feature selection (top 13 features)
- apply select by correlation (threshold 0.9)
- apply target scaling with standard scaler
- Apply resampling keep healthy fraction around 0.4
- Prefer seq2val architecture e.g. lstm, gru etc.
- Don't use tsfresh or flatten based feature extraction

## Feature Engineering Recommendations
- Max window size around 30-100
- Max stride around 1-10

## Regression Model Recommendations
- Seq2Val Model (GRU) works well
- don't use multi window approach