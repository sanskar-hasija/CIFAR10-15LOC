import tensorflow as tf 

(x_train  , y_train) , (x_test , y_test)  = tf.keras.datasets.cifar10.load_data()
x_train = x_train /255 ; x_test = x_test/255

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,
                                                                 zoom_range = 0.05,
                                                                 width_shift_range = 0.05 ,
                                                                 height_shift_range = 0.05)
train_generator = data_generator.flow(x_train,
                                      y_train,
                                      batch_size = 32)

base_model = tf.keras.applications.ResNet50V2(include_top = False,
                                              weights = "imagenet",
                                              input_shape = x_train.shape[1:] , 
                                              classes = 10)
model= tf.keras.Sequential()
model.add(base_model) 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024,activation=('relu'))) 
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Dense(10,activation=('softmax')))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

history = model.fit(train_generator , 
                    validation_data=(x_test , y_test), 
                    steps_per_epoch = x_train.shape[0]/32,
                    epochs=20 ,
                    verbose = 1 )

model.save("CIFAR-10.h5")