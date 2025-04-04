#!/bin/bash
echo Copy data
python split_data.py
echo Preprocess data
python prepare_data_reason.py --data_split train
python prepare_data_reason.py --data_split val
