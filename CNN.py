#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPool2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.optimizers import Adam ,RMSprop ,SGD


# In[6]:


from keras.models import Sequential
model = Sequential()


# In[7]:


import random


# In[8]:


model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu',
                        input_shape=(64, 64, 3)
                       ))
model.add(MaxPool2D(pool_size=(2,2)))


# In[9]:


def architecture(case):
    if case == 1:
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
        model.add(MaxPool2D(pool_size=(2, 2)))
    elif case == 2:
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
        model.add(MaxPool2D(pool_size=(2, 2)))


# In[10]:


architecture(random.randint(1,2))


# In[11]:


model.add(Flatten())


# In[12]:


def fC(case):
    if case == 1:
        model.add(Dense(units=128, activation='relu'))  
    elif case == 2:
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu')) 
    elif case == 3:
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))


# In[13]:


fC(random.randint(1,3))


# In[14]:


model.add(Dense(units=1,activation='sigmoid'))


# In[15]:


print(model.summary())


# In[16]:


lR=.001


# In[17]:


model.compile(optimizer=Adam(lr=lR),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[18]:


lR


# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# In[20]:


model.optimizer


# In[21]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

out = model.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=25,
        validation_data=test_set,
        validation_steps=28)


# In[ ]:


out.history


# In[ ]:


print(out.history['val_acc'][24])


# In[ ]:


optimizer=model.optimizer
learningRate=lR
mod =str(model.layers)
accuracy = str(out.history['val_acc'][24])


# In[ ]:


if out.history['val_acc'][24] >= .78:
    import smtplib
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("sdhah1999@gmail.com", "axnxovombplymgvw")
   


    subject1= 'optimizer'
    subject2= 'learning rate'

    subject3= 'optimizer'
    subject4= 'learning rate'
    
    body1= accuracy
    body2= mod
    
    body3= optimizer
    body4= learningRate

    message1 = f'Subject:{subject1}\n\n{body1}'
    message2 = f'Subject:{subject2}\n\n{body2}'
    message3 = f'Subject:{subject3}\n\n{body3}'
    message4 = f'Subject:{subject4}\n\n{body4}'
    
    
    s.sendmail("sdhah1999@gmail.com", "sdhah1999@gmail.com", message1)
    s.sendmail("sdhah1999@gmail.com", "sdhah1999@gmail.com", message2)
    s.sendmail("sdhah1999@gmail.com", "sdhah1999@gmail.com", message3)
    s.sendmail("sdhah1999@gmail.com", "sdhah1999@gmail.com", message4)
    s.quit()

