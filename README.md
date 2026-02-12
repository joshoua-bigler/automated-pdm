# Automated Modeling for Predictive Maintenance: Comparison and Implementation of AutoML, Agent-Based and LLM-Based approaches for RUL Prediction
This repository contains the implementation and experiments for a master’s thesis on automated remaining useful life (RUL) prediction. It evaluates classical AutoML, LLM-based agentic approaches, and a hybrid agentic AutoML framework on the C-MAPSS and FEMTO datasets.

For more in-depth documentation, see the master’s thesis documentation: [**Master’s Thesis**](masters_thesis_jbi_v1.0.2.pdf).

The work systematically compares three modeling paradigms:

- Classical AutoML  
- Standalone Agentic AI (LLM-based pipeline generation)  
- Hybrid Agentic + AutoML approach  

All approaches are evaluated on:

- NASA C-MAPSS turbofan engine dataset  
- FEMTO (PRONOSTIA) bearing dataset  


## Objective

Predictive maintenance systems require significant machine learning expertise for:

- Data preprocessing  
- Feature engineering  
- Model selection  
- Hyperparameter optimization  

This project investigates:

1. Whether classical AutoML is sufficient for RUL prediction  
2. Whether agentic AI can autonomously generate competitive pipelines  
3. Whether a hybrid approach combines robustness and flexibility  

Evaluation focuses on:

- Predictive performance (RMSE)  
- Robustness and failure behavior  
- Generalizability across datasets  
- Practical deployment effort  

# Software Architecture
The software architecture consists of three coordinated packages: `rul_lib`,
`auto_rul`, and `agent_rul`.  
`rul_lib` provides shared core functionality for data handling, preprocessing,
feature engineering, model training, and evaluation.  

`auto_rul` implements the standalone AutoML workflow with hyperparameter
optimization, while `agent_rul` realises the multi-agent system for automated
pipeline configuration, reasoning, and model generation.  

![Software Architecture](/images/sw_architecture.svg)

## Repository Structure

The project consists of three coordinated packages located directly in this repository.

### `rul_lib`

Core shared functionality:

- Data loading  
- Preprocessing  
- Feature engineering  
- Sliding and multi-window processing  
- Model training  
- Evaluation metrics  
- Utility functions  

This package forms the foundation for both AutoML and agentic execution modes.



### `auto_rul`

Implements the standalone AutoML workflow:

- Structured search spaces  
- Hyperparameter optimization  
- Bayesian optimization (TPE)  
- ASHA scheduler  
- Experiment logging  



### `agent_rul`

Implements the multi-agent system:

- Dataset-aware reasoning  
- Dynamic pipeline generation  
- YAML configuration synthesis  
- Regression model generation  
- Hybrid and standalone agentic execution modes  



## Execution Modes

The system supports three execution strategies:

| Mode                | Description                                              |
|---------------------|----------------------------------------------------------|
| AutoML              | Predefined search space + SMBO-based HPO                |
| Hybrid              | Agent-generated configuration + AutoML optimization     |
| Standalone Agentic  | Fully agent-generated pipeline without HPO              |


# Setup Instructions

### Install Required Dependencies
To run the three different approaches, install the required dependencies from
`/auto_rul`,  `/agent_rul` and `/rul_lib`.  

```bash
pip install ./auto_rul
pip install ./agent_rul
pip install ./rul_lib
```

### Environment Variables
The following environment variables must be set before running the pipeline.
Create a `.env` file in the project root and adjust the values as needed.

### Credentials

- [OpenAI API key](https://platform.openai.com/api-keys)

- [Langfuse keys](https://cloud.langfuse.com)
```env
OPENAI_API_KEY=''
LANGFUSE_SECRET_KEY=''
LANGFUSE_PUBLIC_KEY=''
LANGFUSE_HOST='https://cloud.langfuse.com'
RAY_TMPDIR=''
RAY_SPILL_DIR=''
OPENBLAS_NUM_THREADS='64'
```

## Datasets
In order to run the experiments, download the datasets and place them in the `/data` folder.
- [FEMTO bearing dataset](https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip)
- [NASA C-MAPSS dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

## Services
To enable TensorBoard logging and the MLflow model registry, start the required services using Docker Compose:
```bash
docker compose up -d
```

# Running Experiments
Experiments can be executed via the `auto_rul` and `agent_rul` entry points.
Configuration is controlled through YAML files located in the
`pipeline_configs/` directory of the corresponding package.
