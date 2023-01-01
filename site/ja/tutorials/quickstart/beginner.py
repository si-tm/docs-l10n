import tensorflow as tf

# tensorflow                   2.9.1
# tensorflow-estimator         2.9.0
# tensorflow-io-gcs-filesystem 0.26.0

def load_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 # サンプルデータを整数から浮動小数点数に変換

    return (x_train, y_train), (x_test, y_test)

def make_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'), # units=128 : 出力空間の次元数
        tf.keras.layers.Dropout(0.2), # 入力にドロップアウトを適用する rate=0.2 : 入力ユニットをドロップする割合
        tf.keras.layers.Dense(10) 
        ])

    return model

def predict(model, x_train):
    predictions = model(x_train[:1]).numpy()
    return predictions

def soft_max(predictions):
    return tf.nn.softmax(predictions).numpy()

def lost_function(predictions, y_train):
    loss_fn = \
        tf.keras.losses.SparseCategoricalCrossentropy( 
            from_logits=True 
            ) # 交差エントロピー誤差を計算する, logits : ソフトマックス活性化関数に通す前のニューラルネットワークの出力
    return loss_fn

def model_fit(model, x_test, y_test, x_train, y_train):
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
        ])
    print(probability_model(x_test[:5]))

def main():
    print("looad minst")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print("make model")
    model = make_model()
    print("predict")
    predictions = predict(model, x_train)
    soft_max(predictions)
    print("prepare lost function")
    loss_fn = lost_function(predictions, y_train)
    model.compile(optimizer='adam', # adam最適化　https://keras.io/api/optimizers/
              loss=loss_fn, 
              metrics=['accuracy']) # metrics : 評価関数のリスト
    model_fit(model, x_test, y_test, x_train, y_train)

if __name__ == '__main__':
  main()