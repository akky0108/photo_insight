import tensorflow as tf

# GPUが利用可能か確認
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUは{len(gpus)}台利用可能です。")
else:
    print("GPUは利用できません。")
