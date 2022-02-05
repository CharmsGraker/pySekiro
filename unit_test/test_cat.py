import tensorflow as tf
def test():
    action = tf.convert_to_tensor([0, 1, 5])
    a = tf.keras.utils.to_categorical(action)
    print(a)

if __name__ =='__main__':
    test()
