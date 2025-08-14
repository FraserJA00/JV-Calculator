# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:18:04 2025

@author: Fraser Angus (PGR)
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from jvtool.gui import run_app

if __name__ == "__main__":
    run_app()