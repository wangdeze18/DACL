# t-test and effect size

We add t-test and effect size on code clone task in the latest draft.



## Pipeline

We provides our implementation to do t-test and effect size  on code clone  task.


### run


```Shell
python t_test_main.py
```


## Result

For code clone, the result show that the improvement of GraphCodeBert are significant with p<0.001 and effect size=0.5.

TODO: We will add more baselines results on this task.

| Model                                       |    accuracy    |  
| ------------------------------------------- | :--------: |
| CodeBert                             |  **0.985** |
| GraphCodeBert                             |  **0.987** |
| p-value                             |  **<0.001** |
| effect size                            |  **0.5** |
