**Introduction**
*This project targets problems of insufficient data in text classification tasks. By using some few-shot learning tricks (ProtoNet, etc.), performance on tasks sees improvement, and has potential to furthur improve, but the convergence speed for ProtoNet+bert is much slower than normal Bert finetuning, and GPU memory is also a key limitation on its improvement (cannot set large number of supports at evaluation time, #TODO to fix this in the future)*

**Classification Models**
1. ProtoNet+Bert (optimized for fewshot, can achieve better performance on some small dataset)
2. Ordinary Bert classification (for normal dataset, also works for fewshot thanks to the strength of BERT pretraining)
3. A Mysterious Algorithm from my colleague (optimized for matching tasks, do not train this for normal classification tasks, just for experimental purporse, just for fun)


**Usage:**
1. put your data into ./data folder
2. write your own script (or use some pre-given function in data_formatter.py)
    to format your training/evaluation data into "sentence and its label separated by tab" per line
3. modify configuration in conf/config.py under the Config class for your chosen model,
    *  Mandatory settings:
        *  for Bert classifier: set number of classes and max sentence length,
        *  for ProtoNet: set "k" and "shot", k must be between 20% to 100% of total number of classes, shot commonly between 2 and 10 depending on datasize
    *  Optional settings:
        *  for Bert classifier: batch_size
        *  for ProtoNet: n_support, eval_n_support (number of supporting samples for each class, read the paper on ProtoNet for more details),
            you can just leave them unchanged, the bigger the better, but may exceeds GPU memory limits, 
            especially at evaluation time, when number of classes is big.
        *  general settings: learning rate, warmup, paths to essential data/modelfiles, device, etc..
3. Alternatively, if you are sick of modifying the config file, or you want to train multiple models with different configs, you can just use <*python scripts/api.py*> directly,
all kinds of settings can be re-defined here, overriding what's in config.py. type <*python scripts/api.py -h*> for more details.
4. choose to run from three shell script on your demand
5. predict with the other three shell scripts, don't forget to check all kinds of load paths before running

**Requirements:**
pytorch, transformers, pytorch_pretrained_bert, keras, sklearn, etc..

**Note:**
Recommended hyperparameters are left as they are in conf/config.py except those that are task specific. All experiments are using bert-chinese-base, not tested for other languages, but you can always try it (remember to change bert_type in config).

**TODO:**
1. support unlimited number of supports at evaluation/prediction time
2. support Meta-Learning
3. replace Euclidean distance with RE2 and BCEloss


**-------------------------------------------------------------------------------------------------------:**




|        | ProtoNet+Bert | Bert | Training size| Test size | Balanced | Class Count|
| ------ | ------ |------ |------ |------ |------ |------ |
| Intent Classification (downsampled to 1%)  | **88.3%** | 84.9% | ≈60*15 | 1333 | True | 15 |
| Intent Classification | >93.7%(too low to train) | **94.6%**(?) | ≈6000*15 | 1333 | True | 15|
| Anonymous Dataset 1 | **87.8%** | 87.2% | 3200 | 352 | False | 86|
| Anonymous Dataset 2 | **84.9%** | 84.3% | 1300 | 434 | False | 20|

