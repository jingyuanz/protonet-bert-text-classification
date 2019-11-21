**Classification Models**
1. ProtoNet+Bert (optimized for fewshot, can achieve better performance on some small dataset)
2. An algorithm from Shenyuan (optimized for matching tasks, do not train this for normal classification tasks, just for experimental purporse)
3. Ordinary Bert classification (for normal dataset, also works for fewshot thanks to the strength of BERT pretraining)


**Usage:**
1. put your data into ./data folder
2. write your own script (or use some pre-given function in data_formatter.py)
    to format your training/evaluation data into "[sentence]\t[label]" per line
3a. modify configuration in conf/config.py under the Config class for your chosen model,
    *  Mandatory settings:
        *  for Bert classifier: set number of classes and max sentence length,
        *  for ProtoNet: set "k" and "shot", k must be between 20% to 100% of total number of classes, shot commonly between 2 and 10 depending on datasize
    *  Optional settings:
        *  for Bert classifier: batch_size
        *  for ProtoNet: n_support, eval_n_support (number of supporting samples for each class, read the paper on ProtoNet for more details),
            you can just leave them unchanged, the bigger the better, but may exceeds GPU memory limits, 
            especially at evaluation time, when number of classes is big.
        *  general settings: learning rate, warmup, paths to some essential data/modelfiles
3b. Alternatively, if you are sick of modifying the config file, or you want to train multiple models with different configs, you can just use <*python scripts/api.py*> directly,
all kinds of settings can be re-defined here, overriding what's in config.py. type <*python scripts/api.py -h*> for more details.
4. choose to run from three shell script on your demand
5. predict with the other three shell scripts, don't forget to check all kinds of load paths before running





|        | ProtoNet+Bert | Bert | Datasize | Balanced | Class Count|
| ------ | ------ |------ |------ |------ |------ |
| Intent Classification (downsampled to 1%)  | **80.8%** | 77.9% | ≈60*15 | True | 15 |
| Intent Classification | >93.7%(too low to train) | **94.6%**(?) | ≈6000*15 | True | 15|
| Performance Evaluation Classification | 86.9% | **87.2%** | 3200 | False | 86|
| Key Experience Classification | **84.9%** | 84.3% | 1300 | False | 20|

