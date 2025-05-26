# chromosome-classification
This repository contains my experiments, including trial-and-error approaches with popular and available neural network architectures, for tackling the chromosome classification problem.

# Project structure
```
  pytorch-template/
  │
  ├── main.py - main script to run training and testing
  ├── configs/ - directory for storing config
  ├── parse_config.py - class to handle config file and cli options
  │
  │
  ├── src/
  │   ├── data/ - default directory for storing input data
  │   ├── model/ - models, losses, and metrics
  │   ├── train.py - script to start training
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```