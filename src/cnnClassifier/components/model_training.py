from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf
# ✅ Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()
import os
import urllib.request as request
from zipfile import ZipFile
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """
        Load the saved model but avoid restoring the old optimizer state.
        Then create a new optimizer instance and compile the model for further training.
        """
        try:
            # Attempt to load model (use str path). Do NOT restore old optimizer state -> compile=False
            self.model = tf.keras.models.load_model(
                str(self.config.updated_base_model_path),
                compile=False
            )
        except TypeError as e:
            # If we hit the InputLayer 'batch_shape' problem from earlier, use patched InputLayer
            if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
                class PatchedInputLayer(tf.keras.layers.InputLayer):
                    @classmethod
                    def from_config(cls, config):
                        if 'batch_shape' in config and 'batch_input_shape' not in config:
                            config = dict(config)
                            config['batch_input_shape'] = config.pop('batch_shape')
                        return super().from_config(config)

                self.model = tf.keras.models.load_model(
                    str(self.config.updated_base_model_path),
                    custom_objects={"InputLayer": PatchedInputLayer},
                    compile=False
                )
            else:
                raise

        # Create a new optimizer instance (fresh, not reused)
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # Decide loss depending on generator class_mode (default to categorical)
        loss = "categorical_crossentropy"
        try:
            cm = getattr(self, "train_generator", None)
            if cm is not None and getattr(cm, "class_mode", None) is not None:
                cmode = cm.class_mode
                if cmode == "binary":
                    loss = "binary_crossentropy"
                elif cmode == "sparse":
                    loss = "sparse_categorical_crossentropy"
                else:
                    loss = "categorical_crossentropy"
        except Exception:
            loss = "categorical_crossentropy"

        # Compile model with new optimizer and chosen loss
        self.model.compile(optimizer=new_optimizer, loss=loss, metrics=["accuracy"])

        print(f"Loaded model from {self.config.updated_base_model_path} and compiled with optimizer={type(new_optimizer).__name__}, loss={loss}")

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
