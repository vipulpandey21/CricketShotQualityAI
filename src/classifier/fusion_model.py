"""
fusion_model.py
Multi-modal fusion: RGB frames + Skeleton keypoints
"""

import tensorflow as tf
# Use keras from tensorflow
keras = tf.keras
Input = keras.Input
Model = keras.Model
layers = keras.layers
EfficientNetB0 = keras.applications.EfficientNetB0


def build_fusion_model(num_classes=10):
    """
    Build fusion model with two inputs:
    - RGB frames: (30, 224, 224, 3)
    - Skeleton: (30, 13, 3)
    """
    
    # Branch 1: Visual features (RGB)
    input_rgb = Input(shape=(30, 224, 224, 3), name='rgb_frames')
    base_cnn = EfficientNetB0(include_top=False, weights='imagenet', 
                              input_shape=(224, 224, 3))
    base_cnn.trainable = False
    
    x1 = layers.TimeDistributed(base_cnn)(input_rgb)
    x1 = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x1)
    x1 = layers.GRU(256, return_sequences=True)(x1)
    visual_features = layers.GRU(128, name='visual_features')(x1)  # 128-dim
    
    # Branch 2: Pose features (Skeleton)
    input_skeleton = Input(shape=(30, 13, 3), name='skeleton_keypoints')
    x2 = layers.Reshape((30, 39))(input_skeleton)  # Flatten: 13×3 = 39
    x2 = layers.Dense(128, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    pose_features = layers.LSTM(64, name='pose_features')(x2)  # 64-dim
    
    # Fusion with attention weights
    visual_weight = layers.Dense(1, activation='sigmoid', name='visual_weight')(visual_features)
    pose_weight = layers.Dense(1, activation='sigmoid', name='pose_weight')(pose_features)
    
    # Weighted features
    weighted_visual = layers.Multiply()([visual_features, visual_weight])
    weighted_pose = layers.Multiply()([pose_features, pose_weight])
    
    # Concatenate
    combined = layers.concatenate([weighted_visual, weighted_pose])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax', name='shot_class')(combined)
    
    model = Model(inputs=[input_rgb, input_skeleton], outputs=output)
    return model


def get_weightage(model, rgb_input, skeleton_input):
    """
    Calculate contribution weightage of RGB vs Skeleton
    Returns: (visual_weight, pose_weight)
    """
    # Get intermediate layer outputs
    visual_weight_layer = model.get_layer('visual_weight')
    pose_weight_layer = model.get_layer('pose_weight')
    
    # Create temporary models
    visual_model = Model(inputs=model.input, outputs=visual_weight_layer.output)
    pose_model = Model(inputs=model.input, outputs=pose_weight_layer.output)
    
    # Predict weights
    v_weight = visual_model.predict([rgb_input, skeleton_input], verbose=0)
    p_weight = pose_model.predict([rgb_input, skeleton_input], verbose=0)
    
    # Average across batch
    v_avg = float(v_weight.mean())
    p_avg = float(p_weight.mean())
    
    # Normalize to sum to 100%
    total = v_avg + p_avg
    v_pct = (v_avg / total) * 100
    p_pct = (p_avg / total) * 100
    
    return v_pct, p_pct
