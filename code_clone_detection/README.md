# Code Clone Detection

1. Datasets


   Download the [dataset](https://zenodo.org/record/5376257#.YTC3oI4zZsY) and place it in this folder.

```
mkdir ./code_clone_detection/dataset
```


2. Fine-tune pre-trained models

   You can also skip this step by downloading the [pre-trained model](https://zenodo.org/record/5414294#.YTIb64gzY2w) directly.
```
python run_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_train --train_data_file=../dataset/large_train.txt --eval_data_file=../dataset/valid.txt --test_data_file=../dataset/test.txt --epoch 10 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --pacing_function linear --seed 123456 2>&1| tee train.log;
```

3. Evaluate pre-trained models on the original dataset
```
python run_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.txt --eval_data_file=../dataset/valid.txt --test_data_file=../dataset/test0.txt --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee test.log;
```

4. Test-time augmentation
```
python run_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.txt --eval_data_file=../dataset/valid.txt --test_data_file=../dataset/test0.txt --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee test0.log;
python run_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.txt --eval_data_file=../dataset/valid.txt --test_data_file=../dataset/test1.txt --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee test1.log;
python run_curri.py --output_dir=./saved_models --model_type=roberta --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=roberta-base --do_test --train_data_file=../dataset/large_train.txt --eval_data_file=../dataset/valid.txt --test_data_file=../dataset/test2.txt --epoch 2 --block_size 400 --train_batch_size 128 --eval_batch_size 256 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee test2.log;
```
   #Combine the above results and make predictions
```
python sumscores.py
```
