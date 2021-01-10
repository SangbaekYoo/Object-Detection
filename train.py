# -*- coding: utf-8 -*-
import time
import datetime
import pandas as pd
from glob import glob
from argparse import ArgumentParser

# Import seq2seq trainer
from yolo.trainer import Trainer

# Import global variables
from utils.global_variables import new_size, grid, img_w, img_h, segment, confindency_threshold

if __name__ == '__main__':
    # Argument handling
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50, help='train epochs')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    args = parser.parse_args()

    # Load train data
    label_data = pd.read_csv('./dataset/label_data.csv').values
    num_data = pd.read_csv('./dataset/num_data.csv').values

    # Set trainer
    trainer = Trainer(batch=args.batch, dataset=(label_data, num_data))

    # Train epochs
    print(f'Total Epochs: {args.epochs} | Batch Size: {args.batch}')
    for epoch in range(args.epochs):
        print()
        print(f'EPOCH {epoch + 1} / {args.epochs} : Start at {datetime.datetime.now()}')
        start = time.time()
        loss = trainer.train_iter()
        print(f'| Loss: {loss}')
        print(f'| Epoch Time Taken: {time.time() - start} seconds')
        trainer.save()
