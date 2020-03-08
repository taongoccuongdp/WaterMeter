import tensorflow as tf
from build_model import ModelBuilder
from dataset import DataGenerator
from config import Config
from tensorflow.python.keras.optimizers import Adam
import functools
def train():
    model = ModelBuilder.buid()
    opt = Adam(Config.ADAM_LR)
    train_generator = DataGenerator(True)
    valid_generator = DataGenerator(False)
    model.summary()
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])
    model.fit_generator(generator=train_generator.batch_generator(), 
                        steps_per_epoch=int(train_generator.dataset_size/Config.BATCH_SIZE),
                        epochs=Config.EPOCHS,
                        validation_data= valid_generator.batch_generator(),
                        validation_steps = int(valid_generator.dataset_size/Config.BATCH_SIZE))
    model.save("easy_samples.h5")
if __name__ == '__main__':
    train()