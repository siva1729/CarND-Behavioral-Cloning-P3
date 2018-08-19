
import csv, os
import cv2
import numpy as np
import sklearn
from keras.models import Model
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Function: Read/parse the driving logfile
def read_lines(log_file):
    lines = []
    with open (log_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines

 
# Generator function to read the image data and return the batch_size
# Also Augment the data for training data
def data_generator(samples, img_dir, batch_size=32, valid=0):
    num_samples = len(samples)
    dbg_count = 0
    img_cnt = 0
    angle_correction = 0.2
    while 1:
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images = []; steer_angle = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                img_cnt += 1
                for i in range(3):
                    file_name = img_dir + r'\\' + batch_sample[i].split('\\')[-1]
                    image = cv2.imread(file_name)
                    images.append(image)
                    # For training only
                    # Use data from Left/Right Images with steering angle correction factor
                    if (i == 1):   angle = angle + angle_correction
                    elif (i == 2): angle = angle - angle_correction
                    steer_angle.append(angle)
                    if (valid==1): break
                    # For training only
                    # Data Augmentation to help driving steer Car clockwise direction
                    # Flip the images horizontally and change polarity of the steering angle
                    images.append(cv2.flip(image,1))
                    steer_angle.append(angle * -1.0)
                    if (img_cnt==200 and i==0): 
                        cv2.imwrite('flip_small.png', cv2.flip(image,1))
                        cv2.imwrite('Normal_small.png', image)
                    elif (img_cnt==200 and i==1): 
                        cv2.imwrite('Normal_right.png', image)
                    elif (img_cnt==200 and i==2): 
                        cv2.imwrite('Normal_left.png', image)
                    
            X_train = np.array(images)
            y_train = np.array(steer_angle)  
            if (dbg_count < 3):
                if (valid==1): 
                    print ("Valid_Data # images, labels:",  X_train.shape, len(y_train))
                else:      
                    print ("Train_Data # images, labels:",  X_train.shape, len(y_train))
            dbg_count += 1
            yield sklearn.utils.shuffle(X_train, y_train)

 
## Function for Plotting the loss vs epoch  
def print_plots(history_object):
    ### print the keys contained in the hstory object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot (history_object.history['loss'])
    plt.plot (history_object.history['val_loss'])
    plt.title ('Model MSE loss')
    plt.ylabel ('MSE loss')
    plt.xlabel ('epoch')
    plt.legend(['training_set', 'validation_set'], loc = 'upper right')
    plt.show()

	
### Network Model
import timeit
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# Read/Parse the image input data
log_file1 = 'C:\data4\driving_log.csv'
img_folder1 = 'C:\data4\IMG'
lines1 = read_lines(log_file1)

# Split the dataset into train/validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split (lines1, test_size=0.2)

# Compile and train the model using the generator function
train_generator = data_generator(train_samples, img_folder1, batch_size = 32)
valid_generator = data_generator(validation_samples, img_folder1, batch_size = 32, valid=1)

# Model acrhitecture (Modified CNN Model)
start_time = timeit.default_timer()
print ("start_time:", start_time, "seconds")

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and train the model using mean-squared-error loss and adam optimizer
model.compile (loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator (train_generator, samples_per_epoch=len(train_samples)*6,
                      validation_data=valid_generator,
                      nb_val_samples=len(validation_samples), nb_epoch = 5, verbose = 1)

# Save the model
model.save('model.h5')
print ("Model saved model.h5");
elapsed_time = timeit.default_timer() - start_time
print ("elapsed_time: ", elapsed_time, "seconds")


### Train using the saved model

# Load the existing the model 
#del model

# Read/Parse the image input data++++++++++++++++++++++++++++++++++++++++


log_file2 = 'C:\data4\driving_log.csv'
img_folder2 = 'C:\data4\IMG'
lines2 = read_lines(log_file2)

# Split the dataset into train/validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split (lines2, test_size=0.2)

# Compile and train the model using the generator function
train_generator = data_generator(train_samples, img_folder2, batch_size = 32)
valid_generator = data_generator(validation_samples, img_folder2, batch_size = 32, valid=1)

# Load the saved model
model = load_model('model_good.h5')

# compile and train the model
start_time = timeit.default_timer()
print ("start_time:", start_time, "seconds")
model.compile (loss = 'mse', optimizer = 'adam')
history_object2 = model.fit_generator (train_generator, samples_per_epoch=len(train_samples)*6,
                  validation_data=valid_generator,                      
				  nb_val_samples=len(validation_samples), nb_epoch = 5, verbose = 1)

# Save the model
model.save('model1.h5')
elapsed_time = timeit.default_timer() - start_time
print ("elapsed_time: ", elapsed_time, "seconds")


