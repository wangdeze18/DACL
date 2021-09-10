# Algorithm Classification

1. Datasets


    Download the [dataset](https://zenodo.org/record/5376257#.YTC3oI4zZsY) and place it in this folder.

```
mkdir ./algorithm_classfication/dataset
```


2. Fine-tune pre-trained models

   You can also skip this step by downloading the [pre-trained model](https://zenodo.org/record/5414294#.YTIb64gzY2w) directly.
   
   NOTE: The `large_class_score.npy` is calculated by cross-validation experiments, and we also provide a ready-to-use version.
```
python run_class_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_train --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl --epoch 20 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function root_10 --class_score large_class_score.npy --seed 123456 2>&1| tee train.log;
```

3. Evaluate pre-trained models on the original dataset
```
python run_class_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function root_10 --seed 123456 2>&1| tee test.log; 
python ../evaluator/extract_answers.py -c ../dataset/test.jsonl -o saved_models/answers.jsonl; 
python ../evaluator/evaluator.py -a saved_models/answers.jsonl -p saved_models/predictions.jsonl; 
python ../evaluator/eva_MAP.py -a saved_models/answers.jsonl -p saved_models/predictions.jsonl;
```

4. Test-time augmentation
```
python run_class_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test_align1.jsonl --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function root_10 --seed 123456 2>&1| tee test.log; 
python run_class_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test_align2.jsonl --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function root_10 --seed 123456 2>&1| tee test.log; 
python run_class_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test_align3.jsonl --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function root_10 --seed 123456 2>&1| tee test.log;
```
   #Combine the above results and make predictions
   
```
python bagging.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee test.log;
python ../evaluator/extract_answers.py -c ../dataset/test.jsonl -o saved_models/answers.jsonl; 
python ../evaluator/evaluator.py -a saved_models/answers.jsonl -p saved_models/predictions.jsonl  2>&1| tee evaluator.log; 
python ../evaluator/eva_MAP.py -a saved_models/answers.jsonl -p saved_models/predictions.jsonl  2>&1| tee eva_MAP.log;
```

