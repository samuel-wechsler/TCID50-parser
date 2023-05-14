import tensorflow as tf

temp = tf.Variable(intial_value=1.0, trainable=True, dtype=tf.float32)


def compute_loss(Y, y_pred):
    y_pred_model_w_temp = tf.math.divide(y_pred, temp)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        tf.convert_to_tensor(tf.keras.utils.to_categorical(Y))
    ))

    return loss


def optimize():
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    print(f"Temperature Initial value: {temp.numpy()}")

    for i in range(300):
        opts = optimizer.minimize(compute_loss, var_list=[temp])

    print(f"Temperature Final value: {temp.numpy()}")
