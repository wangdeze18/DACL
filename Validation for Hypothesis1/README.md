# Data

Datasets (D' and multiple test sets) and the pre-trained model (fine-tuned by D') can be download [here](https://zenodo.org/record/5679348#.YY4s9GBBxsY).

# Results

| Results    |     Original | 1-Trans | 2-Trans | 3-Trans | 4-Trans | 5-Trans |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| Precision   |      0.9295   |     0.9122 |  0.8912 |  0.8763 |  0.8599 | 0.8494 |
| MAP@R     |      0.9094   |     0.8888 |  0.8638 |  0.8461 |  0.8267 | 0.8142 |


See log file for details.


# Illustration

We trained CodeBERT on the entire augmented dataset D' and evaluated on its sub-datasets(Original, 1-Trans, 2-Trans, 3-Trans, 4-Trans and 5-Trans). These datasets are equal in size to ensure the fairness of the experiment.

The experimental results reveal that as the number of transformations increases, the learning effects of the dataset become worse.







