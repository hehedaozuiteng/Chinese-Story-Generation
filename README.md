# Dependency and Semantic Denoising On Chinese Story Generation Task

## 1. Introduction

This is a code repository for the improving the performance of baseline model from [LOT benchmark](https://github.com/thu-coai/LOT-LongLM) on its Chinese Story Generation Task.

## 2. Baseline Model

1. **Download:** The checkpoints of baseline model and example data can be downloaded from [THUCloud](https://cloud.tsinghua.edu.cn/d/576f340a43964a23b1a5/) or [Hugging Face Model Card](https://huggingface.co/thu-coai). The training and generation scripts are under the directory `LOT-LongLM\longlm`.

2. **Model Loading:**

   ```python\
   from transformers import T5Tokenizer, T5ForConditionalGeneration
   tokenizer = T5Tokenizer.from_pretrained('thu-coai/LongLM-base')
   model = T5ForConditionalGeneration.from_pretrained('thu-coai/LongLM-base')
   ```

3. **Training:**

   Execute `bash ./finetune.sh` to fine-tune LongLM. If deepspeed is available, you can execute `bash ./finetune_deepspped.sh` to accelerate. You can also use the [official script](https://github.com/huggingface/transformers/tree/v4.6.0-release/examples/legacy/seq2seq) provided by Transformers to fine-tune the model.

   ```shell
   env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
   finetune_trainer.py \
   --data_dir=./data \ # directory of data
   --train_name=train \ # file prefix of the training data
   --output_dir=./save_model \ # output directory to save the checkpoint
   --save_total_limit=10 \ # maximum number of the saved checkpoints
   --per_gpu_train_batch_size=3 \ # batch size for training
   --per_gpu_eval_batch_size=3 \ # batch size for evaluation
   --num_train_epochs=1 \ # number of training epochs
   --logging_steps=5 \ # number of stps to log the loss value
   --model_name_or_path=./LongLM-base \ # path to the pretrained model
   --warmup_steps=100 \ # number of steps for warmup
   --learning_rate=1e-4 \ # learning rate
   --n_val=100 \ # number of examples for validation
   --do_train --do_eval \ # whether to training/validation
   --evaluation_strategy steps \ # strategy of evaluation
   --gradient_accumulation_steps=40 # number of steps for gradient accumulation
   --overwrite_output_dir \
   --load_best_model_at_end
   ```

4. **Generation:**

   ```python
   input_ids = tokenizer("小咕噜对，<extra_id_1>",return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
   gen = model.generate(input_ids, do_sample=True, decoder_start_token_id=1, top_p=0.9, max_length=512)
   ```

## 3. Dataset

Data statistics of the OutGen task in LOT. The abbreviations char/sent/len stand for character/sentence/length, respectively.

The datasets and evaluation scripts can be downloaded from [THUCloud](https://cloud.tsinghua.edu.cn/d/0cf033b0c7c049be855d/).

## 4. Ablation Experiment

### Dependency Tagging

### Semantic Denoising

### 5. Dependencies

dependencies files
