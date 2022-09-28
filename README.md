# Dependency and Semantic Denoising On Chinese Story Generation Task

## 1. Introduction

This is a code repository for the improving the performance of baseline model from [LOT benchmark](https://github.com/thu-coai/LOT-LongLM) on its Chinese Story Generation Task.

## 2. Baseline Model

1. **Download:** The checkpoints of baseline model and example data can be downloaded from [THUCloud](https://cloud.tsinghua.edu.cn/d/576f340a43964a23b1a5/) or [Hugging Face Model Card](https://huggingface.co/thu-coai). The training and generation scripts are under the directory `./LOT-LongLM/longlm`.

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

![](figure/datasetTable.PNG)

The datasets and evaluation scripts can be downloaded from [THUCloud](https://cloud.tsinghua.edu.cn/d/0cf033b0c7c049be855d/).

The python script [jsontrans.py](DependencyandSemanticDenoising/jsontrans.py) provides APIs to convert the jsonl file downloaded from THUCloud into the `.source` and `.target` file required for [training script](#Generation).

## 4. Experiments

### Generation Tasks

The training script of LongLM for the generation tasks is the same as pretraining script. The generation script and example data can be found under `./LOT-LongLM/baseline/generation`. You can execute `bash ./gen.sh` for generation.

### Dependency Tagging

The python script [DependencyTagging.py](DependencyandSemanticDenoising/DependencyTagging.py) consumes the `.jsonl` files and adding the dependency tokens into the story, in order to produce the `.source` and `.target` files.

### Semantic Denoising

The python script [boost_simbert.py](DependencyandSemanticDenoising/boost_simbert.py) consumes the `.jsonl` files and expanded data to 6 times the original size, in order to produce the `.source` and `.target` files.

### Dependency and Semantic Denoising

1. Run the python script [boost_simbert.py](DependencyandSemanticDenoising/boost_simbert.py) with the codes :

   ```python

    data = load_file("./boosts_bert/train.jsonl") # read the training data from json file
    data = boost_data(data, gen_synonyms, 5) # Expanded data to 6 times the original size
    write_jsonl_file_source("./boosts_bert/train_new.jsonl", data) #saving the data as jsonal file

   ```

   To generate a new `train.jsonl`

2. Tuns the python script [DependencyTagging.py](DependencyandSemanticDenoising/DependencyTagging.py) to label the dependency tagging.

   ```python

    data = load_file("./boosts_bert/train_new.jsonl")
    # read the training data from json file
    write_txt_file_source("./outgen/train.source",data)
    # save the outline from data to the file
    write_txt_file_target("./outgen/train.target",data)

   ```

3. Training the model.

### 5. Dependencies

Difference script requires different dependencies environments. You can find the different version [requirements.txt](requirements/requirements.txt) in the folder [requirements](requirements).

[Dependencis for training](requirements/requirements_training.txt)

```
datasets                1.6.2
deepspeed               0.3.16
huggingface-hub         0.0.8
jieba                   0.42.1
jsonlines               2.0.0
nltk                    3.5
numpy                   1.19.5
pytorch-lightning       1.2.0
regex                   2020.11.13
rouge                   1.0.1
rouge-score             0.0.4
sacrebleu               1.5.0
scipy                   1.5.4
sentencepiece           0.1.95
tokenizers              0.10.1
torch                   1.8.1
torchaudio              0.8.0
torchmetrics            0.2.0
torchvision             0.9.0
transformers            4.6.1
```

[Dependencis for boost_simbert.py](requirements/requirements_roformer-sim.txt)

```
tensorflow               1.14
keras                   2.3.1
bert4keras             0.10.6

```

[Dependencis for DependencyTagging](DependencyandSemanticDenoising/DependencyTagging.py)

```
hanlp
```

## Citation

```txt
@misc{tang2022CSG,
      title={Improving Chinese Story Generation via Awareness of Syntactic Dependencies and Semantics},
      author={Henglin Huang and Chen Tang and Tyler Loakman and Frank Guerin and Chenghua Lin},
      year={2022}
}
```
