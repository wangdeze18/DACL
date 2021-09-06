# DACL

This repository contains code for paper "Bridging Pre-trained Models and Downstream Tasks for Source Code Understanding".
# Directory Structure
In this repository, we have the following directories:

## ./algorithm_classification


```
./algorithm_classification    # This subdirectory contains code for algorithm classification task.
 + ./code                     # Code for model training
   + ./bagging.py             # Code for test-time augmentation
   + ./model.py               # Code for the architecture of the pre-trained model
   + ./pacing_fucntions.py    # Code to implement pacing functions
   + ./run.py                 # Original Code to train (fine-tune) the pre-trained model
   + ./run_apaptive_curri.py  # Fine-tune the pre-trained model in a general curriculum learning strategy
   + ./run_class_curri.py     # Fine-tune the pre-trained model in the class-based curriculum learning strategy (ours)
 + ./evaluator                # Evaluate the pre-trained model on metrics (Precision and MAP)
```

## ./code_clone_detection


```
./code_clone_detection        # This subdirectory contains code for code clone detection task.
 + ./code                     # Code for model training
   + ./model.py               # Code for the architecture of the pre-trained model
   + ./pacing_fucntions.py    # Code to implement pacing functions
   + ./run.py                 # Original Code to train (fine-tune) the pre-trained model
   + ./run_large.py           # Fine-tune the pre-trained model with large (augmented) dataset without any curriculum learning strategy
   + ./run_curri.py           # Fine-tune the pre-trained model in the augmentation-based curriculum learning strategy (ours)
   + ./sumscores.py           # Code for test-time augmentation
 + ./evaluator                # Evaluate the pre-trained model on metrics (Precision, Recall, and F1)
```

## ./codesearch


```
./codesearch                  # This subdirectory contains code for code search task.
 + ./dataset                  # Contains code to preprocess the dataset
 + ./parser                   # Contains code to extract control flow graph for GraphCodeBERT
 + ./convert_scores.py        # Convert experimental results for test-time augmentation (align the retrieved results according to the urls of the queries)
 + ./model.py                 # Code for the architecture of the pre-trained model
 + ./pacing_fucntions.py      # Code to implement pacing functions
 + ./run.py                   # Original Code to train (fine-tune) and evaluate the pre-trained model
 + ./run_large.py             # Fine-tune the pre-trained model with large (augmented) dataset without any curriculum learning strategy
 + ./run_curri.py             # Fine-tune the pre-trained model in the augmentation-based curriculum learning strategy (ours)
 + ./sumscores.py             # Code for test-time augmentation
```

# Datasets and Models 
**We also provide [datasets](https://zenodo.org/record/5376257#.YTC3oI4zZsY) and [pre-trained models](https://zenodo.org/record/5414294#.YTIb64gzY2w) fine-tuned by our approach to verify the results.**

# How to use
**Detailed documentation is coming soon！**


