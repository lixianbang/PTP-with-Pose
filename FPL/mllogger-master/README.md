# MLLogger: simple utilities for machine learning experiment

## Functionalities
* Auto creation of unique directory
* Otherwise, you can specify directory name as use like by setting init=False
* Arguments save/load

## Dependencies
* Python 2.7+ or 3.4+

## Installation
```
python setup.py install
```

## Quick use

```
$ cd tests/
$ python test_mllogger.py --cond sample.json --lr 0.001
Logger test
{'option': None, 'decay_step': [100, 300, 500], 'cond_dir': '', 'dataset': 'MNIST', 'cond': 'sample.json', 'lr': 0.001, 'momentum': 0.9}
$ ls outputs_test/
yymmdd_HHMMSS
$ cat outputs_test/yymmdd_HHMMSS/log_yymmdd_HHMMSS.txt
2017-12-11 15:50:44,656 [INFO] Logger test
2017-12-11 15:50:44,656 [INFO] {'lr': 0.1, 'dataset': 'MNIST', 'cond': None, 'option': None, 'decay_step': [100, 200], 'momentum': 0.9}
$ cat outputs_test/yymmdd_HHMMSS/args.json
{"--lr": ["0.1", "float"], "--option": ["None", "NoneType"], "--momentum": ["0.9", "float"], "--decay_step": [["100", "200"], "int"], "--dataset": ["MNIST", "str"]}
```
