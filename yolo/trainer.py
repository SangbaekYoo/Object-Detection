import os
import json
import tensorflow as tf

from tqdm import tqdm

from yolo.model import Model_Based_YOLO

class Trainer():
    def __init__(self, batch, dataset):
        self.model = Model_Based_YOLO(img_size=400, grid=4)
        self.batch = batch
        input_train, target_train = dataset
        input_tensors , target_tensors = tf.convert_to_tensor(input_train), tf.convert_to_tensor(target_train)
        self.dataset = tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors)).shuffle(10000)
        self.dataset = self.dataset.batch(batch, drop_remainder=True)
        self.steps_per_epoch = len(input_tensors) // batch

        self.optimizer = tf.keras.optimizers.SGD()
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.checkpoint_path = f'./yolo/checkpoint/'
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        self.train_info = {
            'img_size' : 400,
            'grid' : 4,
            'batch' : self.batch
        }

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Latest Checkpoint Restored.')

    def _train_step(self, input_batch, target_batch):
        with tf.GradientTape() as tape:
            y_batch =self.model(input_batch)
            loss = self.loss_object(target_batch, y_batch)
        batch_loss = loss // int(target_batch.shape[1])
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train_iter(self):
        total_loss = 0
        pbar = tqdm(total=self.steps_per_epoch)
        pbar.set_description(f'| yolo train iter')
        for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
            batch_loss = self._train_step(inp, targ)
            total_loss += batch_loss
            pbar.update(1)
        pbar.close()
        return total_loss

    def save(self):
        self.checkpoint_manager.save()

        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')
        with open(train_info_file_path, 'w') as f:
            json.dump(self.train_info, f, indent=4)



