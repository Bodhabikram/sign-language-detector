import tensorflow as tf

from device import DEVICE, print_device_summary


print_device_summary()

print("\nTesting TensorFlow operation...")
print(f"Using device: {DEVICE}")

with tf.device(DEVICE):
    a = tf.random.normal((2000, 2000))
    b = tf.random.normal((2000, 2000))
    c = tf.matmul(a, b)

print("Matrix multiplication completed.")
print("Result shape:", c.shape)
print("Result device:", c.device)
