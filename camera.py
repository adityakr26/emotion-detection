import cv2




import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from gtts import gTTS
import pygame
import playsound
import tensorflow.keras
from process_labels import gen_labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves

# def speak(text):#this defines the function SPEAK and creates the command for its use
#         tts = gTTS(text=text, lang="en")
#         filename = "voice.mp3"
#         tts.save(filename)
#         playsound.playsound(filename)
#         os.remove(filename) #saves the audio file

def speak(text):#this defines the function SPEAK and creates the command for its use
        tts = gTTS(text=text, lang="en")
        filename = "voicethree.mp3"
        tts.save(filename)
        audio_file = os.path.dirname(__file__) + "/voicethree.mp3"
        playsound.playsound(audio_file)
        os.remove(audio_file)
         #saves the audio file

def pianoF(filename):
	playsound(filename,False)



def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            speak("The man infront of you is" + emotion_dict[maxindex])
            
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()



# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
# image = cv2.VideoCapture(0)
# # Load the model
# model = tensorflow.keras.models.load_model('keras_model.h5')

# """
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1."""
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# # A dict that stores the labels
# labels = gen_labels()

# class Videotwo(object):
#     def __init__(self):
#         self.video=cv2.VideoCapture(0)
#     def __del__(self):
#         self.video.release()
#     def get_frame(self):
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         ret,frame=self.video.read()
#         frame = cv2.flip(frame, 1)
#     # In case the image is not read properly
#         # if not ret:
#         #     continue
            
#         # Draw a rectangle, in the frame
#         frame = cv2.rectangle(frame, (220, 80), (530, 360), (0, 0, 255), 3)
#         # Draw another rectangle in which the image to labelled is to be shown.
#         frame2 = frame[80:360, 220:530]
#         # resize the image to a 224x224 with the same strategy as in TM2:
#         # resizing the image to be at least 224x224 and then cropping from the center
#         frame2 = cv2.resize(frame2, (224, 224))
#         # turn the image into a numpy array
#         image_array = np.asarray(frame2)
#         # Normalize the image
#         normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#         # Load the image into the array
#         data[0] = normalized_image_array
#         pred = model.predict(data)
#         result = np.argmax(pred[0])

#         # Print the predicted label into the screen.
#         cv2.putText(frame,  "Label : " +
#                     labels[str(result)], (280, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        
            
#         ret,jpg=cv2.imencode('.jpg',frame)
#         return jpg.tobytes()