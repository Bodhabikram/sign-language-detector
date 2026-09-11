"""
Central CPU/GPU device configuration
Sign Language Detector

This module:
- Detects available TensorFlow GPUs
- Configures GPU memory growth
- Falls back automatically to CPU
- Provides a single DEVICE variable for the project
- Prints useful hardware information

The rest of the project should import DEVICE from this file
instead of implementing GPU/CPU detection separately.
"""

import os
import tensorflow as tf


# OPTIONAL SETTINGS

# Prevent TensorFlow from unnecessarily creating excessive
# CPU thread pools.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


# GPU DETECTION

GPUS = tf.config.list_physical_devices("GPU")
CPUS = tf.config.list_physical_devices("CPU")


# GPU CONFIGURATION

if GPUS:
    print("=" * 70)
    print("GPU CONFIGURATION")
    print("=" * 70)

    print(f"GPU devices detected: {len(GPUS)}")

    for index, gpu in enumerate(GPUS):
        print(f"GPU {index}: {gpu}")

        try:
            # Allow TensorFlow to allocate GPU memory as required
            # instead of reserving all available GPU memory.
            tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as error:
            print(f"Could not configure memory growth: {error}")

    # Primary GPU
    DEVICE = "/GPU:0"

    DEVICE_TYPE = "GPU"

else:
    print("=" * 70)
    print("GPU CONFIGURATION")
    print("=" * 70)

    print("No TensorFlow-compatible GPU detected.")
    print("TensorFlow will use the CPU.")

    DEVICE = "/CPU:0"

    DEVICE_TYPE = "CPU"


# HARDWARE INFORMATION

print(f"Selected device: {DEVICE}")
print(f"Device type: {DEVICE_TYPE}")
print(f"CPU devices detected: {len(CPUS)}")
print(f"TensorFlow version: {tf.__version__}")

print("=" * 70)


# HELPER FUNCTIONS

def get_device():
    """
    Return the currently selected TensorFlow device.

    Returns:
        str: '/GPU:0' if a GPU is available,
             otherwise '/CPU:0'.
    """
    return DEVICE


def gpu_available():
    """
    Check whether TensorFlow can access a GPU.

    Returns:
        bool: True if GPU is available, otherwise False.
    """
    return len(GPUS) > 0


def get_gpu_count():
    """
    Return the number of GPUs detected by TensorFlow.

    Returns:
        int: Number of detected GPUs.
    """
    return len(GPUS)


def print_device_summary():
    """
    Print a concise hardware summary.
    """
    print("\nDevice Summary")
    print("-" * 40)
    print(f"TensorFlow : {tf.__version__}")
    print(f"Device     : {DEVICE}")
    print(f"Type       : {DEVICE_TYPE}")
    print(f"GPUs       : {len(GPUS)}")
    print(f"CPUs       : {len(CPUS)}")
    print("-" * 40)


# MAIN TEST

if __name__ == "__main__":
    print_device_summary()