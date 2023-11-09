#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <yagi@ks.cs.titech.ac.jp>
#
# Distributed under terms of the MIT license.

import argparse
from mllogger import MLLogger
from arghelper import LoadFromJson
from logging import DEBUG
logger = MLLogger("outputs_test", level=DEBUG, init=False)  # Create outputs/yymmdd_HHMMSS/

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--decay_step', type=int, nargs='+', default=[100, 200])
parser.add_argument('--option', type=str, default=None)
parser.add_argument('--cond', type=str, action=LoadFromJson)
args = parser.parse_args()

logger.initialize()
logger.info('Logger test')
logger.info(vars(args))
save_dir = logger.get_savedir()
logger.save_args(args)
