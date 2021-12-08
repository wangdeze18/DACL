# DACL

This repository contains code for paper "Bridging Pre-trained Models and Downstream Tasks for Source Code Understanding". (Accepted to the 44th International Conference on Software Engineering (ICSE 2022))
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

## ./Validation for Hypothesis1


```
./Validation for Hypothesis1  # This subdirectory contains data and results of the validation for Hypothesis1
 + ./hypothesis_1.log         # The log file of the validation experiment for Hypothesis1
```

## ./data_preparation(4.1)

```
./data_preparation(4.1)       # This subdirectory contains an example program for cleaning POJ104
 + ./to_correct.py            # The example program to remove the non-compliable programs 
```

## ./statistical analysis

```
./statistical analysis        # This subdirectory contains a statistical analysis for code clone detection on BigCloneBench
 + ./labels.npy               # True labels (1 for cloned pairs, 0 for others)
 + ./scorest-graphcodebert.npy# Predictions (logits) made by original GraphCodeBERT
 + ./scorest_codebert.npy     # Predictions (logits) made by the augmented CodeBERT which are fine-tuned by our approach 
 + ./t_test_main.py           # Code for t-test and effect size computation(Cohen's d) 
```

## ./data for Analysis Section

```
./data for Analysis Section   # This subdirectory contains data for Analysis Section (Section 5)
```

# How to use
We apply our approach to code pre-trained models CodeBERT and GraphCodeBERT for fine-tuning. 
  
  Please refer to the subdirectory of tasks for more details!


For general fine-tuning (baseline models), please refer to [algorithm classification](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104), [code clone detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench), and [code search](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch).

# Datasets and Models 
**We also provide [datasets](https://zenodo.org/record/5376257#.YTC3oI4zZsY) and [pre-trained models](https://zenodo.org/record/5414294#.YTIb64gzY2w) fine-tuned by our approach to verify the results.**

# Acknowledgement
Our implementation is adapted from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) and [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch) for the implementation of pre-trained models.

# Reference
If you use this code, please consider citing us.
```
@inproceedings{Wang2021BridgingPM,
  title={Bridging Pre-trained Models and Downstream Tasks for Source Code Understanding},
  author={Deze Wang and Zhouyang Jia and Shanshan Li and Yue Yu and Yun Xiong and Wei Dong and Xiangke Liao},
  year={2021}
}
```
