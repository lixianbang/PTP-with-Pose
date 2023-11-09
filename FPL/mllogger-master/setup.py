#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <yagi@ks.cs.titech.ac.jp>
#
# Distributed under terms of the MIT license.

from distutils.core import setup

setup(name='mllogger',
      version='1.0',
      py_modules=['mllogger', 'arghelper', 'summary_logger'],
      description='Simple logger and argument management for general experiments.',
)
