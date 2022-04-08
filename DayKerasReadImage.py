from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import callbacks

#train_dir = '.\\image\\ysgqxt\\'
#train_dir = '.\\reimageLagerExampleDay_CM_forFit\\'
train_dir = '.\\TrainSet\\'
test_dir = '.\\TestSet\\'
num_epochs = 20
batch_size = 128
xsize = 256
ysize = 256

#data_gen = ImageDataGenerator(rescale=1. / 255,validation_split=0.3)
data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = data_gen.flow_from_directory(train_dir,
                                               target_size=(xsize, ysize),
                                               batch_size=batch_size,
                                               class_mode='categorical')

validation_generator = data_gen.flow_from_directory(test_dir,
                                               target_size=(xsize, ysize),
                                               batch_size=batch_size,
                                               class_mode='categorical')



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(xsize, ysize,3), data_format='channels_first')) 

model.add(MaxPooling2D(pool_size=(2,2)))
print([1,model.output_shape])

model.add(Conv2D(32,(3,3),activation = 'relu'))
print([2,model.output_shape])

model.add(MaxPooling2D(pool_size=(2,2)))
print([3,model.output_shape])

model.add(Dropout(0.25)) 
print([4,model.output_shape])

model.add(Flatten())
print([5,model.output_shape])

model.add(Dense(128,activation='relu'))
print([6,model.output_shape])

model.add(Dropout(0.5))
print([7,model.output_shape])

model.add(Dense(5,activation='softmax'))
print([8,model.output_shape])


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
tensor_board = callbacks.TensorBoard()
model.fit_generator(generator=train_generator,
                    epochs=num_epochs,
                    validation_data=validation_generator,
                    callbacks=[tensor_board])
model.save("rebuildModel.h5")
print('ok')
#score = model.evaluate(X_test,y_test,verbose = 0)
 
