from __future__ import annotations

import os
from argparse import ArgumentParser
from typing import List
from config import DATA

def build_parse():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", help="start mode, train, download_data", metavar="MODE", default="train")
    return parser

def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
            
def main() -> int:
    parser = build_parse()
    options = parser.parse_args()
    check_and_make_directories([])