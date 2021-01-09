import os
import json
import tensorflow as tf

from yolo.model import Model_Based_YOLO

class Module():
    def __init__(self):
        self.checkpoint_path = f'./yolo/checkpoint/'
        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')

        with open(train_info_file_path, 'r') as f:
            train_info = json.load(f)

        self.model =Model_Based_YOLO(img_size=train_info['img_size'], grid=train_info['grid'])
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('YOLO model Latest Checkpoint Restored.')

    def predict(self, x):
        input = tf.expand_dims(x, 0)
        result = self.model(x)
        return result



