import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, CHANNELS, LEARNING_RATE

def build_mobilenetv2_model(num_classes: int, freeze_base: bool = True) -> tf.keras.Model:

    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    
    # 1. Base Pretrained Backbone (MobileNetV2 without top classifier)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze feature extractor layers during initial training
    base_model.trainable = not freeze_base
    
    # 2. Input and Preprocessing layer (scales inputs from [0, 255] to [-1, 1] as required by MobileNetV2)
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mnv2_preprocess")(inputs)
    
    # 3. Backbone Feature Extraction
    x = base_model(x, training=False)
    
    # 4. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(128, activation='relu', name="dense_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    
    # Softmax output layer
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="SignLanguage_MobileNetV2")
    
    # 5. Compile with Adam optimizer & Categorical Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_for_fine_tuning(model: tf.keras.Model, unfreeze_last_n: int = 30, fine_tune_lr: float = 1e-5):

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
            
    if base_model is None:
        return model
        
    base_model.trainable = True
    
    # Freeze all layers except the last unfreeze_last_n layers
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
