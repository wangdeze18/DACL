# Statistical analysis for code clone detection on BigCloneBench

We add a statistical analysis with t-test and effect size computation(Cohen's d) on code clone detection task. We compare CodeBERT fine-tuned by our approach with original GraphCodeBERT. 



## Pipeline

We provides our implementation to do t-test and effect size computation on code clone task.



### run


```Shell
python t_test_main.py
```


## Result

For code clone detection, the results show that the improvement of CodeBERT fine-tuned with our approach over GraphCodeBERT is significant with p<0.001 and Cohen's d = 0.498.

TODO: We will add more statistical analysis results on other tasks.
