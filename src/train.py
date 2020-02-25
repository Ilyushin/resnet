import tensorflow as tf
from src.models import resnet_34, resnet_50
import src.metrics as metrics


model = resnet_34.get_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', metrics.eer]
)

model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/res_net'
)

tf.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
tf.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())

history = model.fit(
    train_dataset,
    epochs=2,
    steps_per_epoch=100,
    callbacks=[tensorboard_callback],
    verbose=1
)

