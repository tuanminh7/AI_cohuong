import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Cau hinh duong dan
DATASET_PATH = 'dataset'
MODEL_PATH = 'plant_model.h5'
DISEASES_JSON = 'diseases_info.json'

print("=" * 60)
print("PHIEN BAN TOI UU CHO WINDOWS")
print("=" * 60)


tf.keras.backend.set_floatx('float32')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Da phat hien {len(gpus)} GPU")
    except RuntimeError as e:
        print(f"GPU warning: {e}")
else:
    print("Chay tren CPU mode (se cham hon)")
    # Toi uu cho CPU
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

def create_optimized_model(num_classes):
    """
    Su dung MobileNetV2 - nhe va toi uu cho CPU/Windows
    """
   
    tf.keras.backend.clear_session()
    
    # MobileNetV2 nhe hon EfficientNet, chay tot tren CPU
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
   
    base_model.trainable = False
    
   
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_pipeline(directory, subset, batch_size=32):
    """
    Data pipeline don gian cho Windows
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        shuffle=True if subset == 'training' else False
    )
    
    return generator

def train_model():
    """
    Training toi uu cho Windows
    """
    print("\n" + "=" * 60)
    print("BAT DAU TRAINING")
    print("=" * 60)
    

    if not os.path.exists(DATASET_PATH):
        print(f"\nLoi: Thu muc '{DATASET_PATH}' khong ton tai!")
        print("Vui long tao thu muc 'dataset' va them cac thu muc con")
        return

    disease_classes = [d for d in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, d))]
    num_classes = len(disease_classes)
    
    if num_classes == 0:
        print("\nLoi: Khong tim thay thu muc benh trong dataset!")
        return
    
    print(f"\nTim thay {num_classes} lop benh:")
    for i, cls in enumerate(disease_classes, 1):
        num_images = len(os.listdir(os.path.join(DATASET_PATH, cls)))
        print(f"  {i}. {cls}: {num_images} anh")
    

    total_images = sum([len(os.listdir(os.path.join(DATASET_PATH, cls))) 
                       for cls in disease_classes])
    print(f"\nTong so anh: {total_images}")
    
    if gpus:
        batch_size = 64
    else:
        batch_size = 16  
    print(f"Batch size: {batch_size}")
    

    print("\n" + "-" * 60)
    print("Dang tai du lieu...")
    print("-" * 60)
    
    train_data = create_data_pipeline(
        DATASET_PATH, 
        'training', 
        batch_size=batch_size
    )
    
    val_data = create_data_pipeline(
        DATASET_PATH, 
        'validation', 
        batch_size=batch_size
    )
    
    print(f"\nTraining samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    

    print("\n" + "-" * 60)
    print("Dang tao mo hinh MobileNetV2...")
    print("-" * 60)
    
    model = create_optimized_model(num_classes)
    print(f"\nTong so parameters: {model.count_params():,}")
    

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Tinh steps
    if gpus:

        steps_per_epoch = min(train_data.samples // batch_size, 50)
        validation_steps = min(val_data.samples // batch_size, 20)
        epochs = 15
    else:

        steps_per_epoch = min(train_data.samples // batch_size, 30)
        validation_steps = min(val_data.samples // batch_size, 10)
        epochs = 10
    
    print(f"\nCau hinh training:")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Validation steps: {validation_steps}")
    print(f"  - Epochs: {epochs}")
    

    print("\n" + "=" * 60)
    print("BAT DAU HUAN LUYEN")
    print("=" * 60)
    print()
    
    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("HOAN THANH TRAINING")
    print("=" * 60)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nKet qua cuoi cung:")
    print(f"  Training accuracy:   {final_train_acc:.2%}")
    print(f"  Validation accuracy: {final_val_acc:.2%}")
    print(f"  Training loss:       {final_train_loss:.4f}")
    print(f"  Validation loss:     {final_val_loss:.4f}")
    
    # Luu mo hinh
    print("\n" + "-" * 60)
    print("Dang luu ket qua...")
    print("-" * 60)
    
    model.save(MODEL_PATH)
    print(f"\n[OK] Mo hinh da duoc luu: {MODEL_PATH}")

    class_indices = train_data.class_indices
    with open('class_indices.json', 'w', encoding='utf-8') as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)
    print(f"[OK] Class indices da duoc luu: class_indices.json")
    

    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"[OK] Training history da duoc luu: training_history.json")
    
    print("\n" + "=" * 60)
    print("TAT CA DA HOAN TAT!")
    print("=" * 60)
    
    return model, history

if __name__ == '__main__':
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nDa dung training!")
    except Exception as e:
        print(f"\n\nLoi: {e}")
        import traceback
        traceback.print_exc()