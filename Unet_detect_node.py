#!/usr/bin/python3
# this script using Unet to train model to detect node
import keras
from keras.layers import Conv2D,MaxPool2D,Input,merge,SpatialDropout2D,UpSampling2D
import itertools
from keras import Model
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ((2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1))

def unet_model(dropout_rate,learn_rate, width):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(inputs)
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(pool1)
    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(pool2)
    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(pool3)
    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="elu")(pool4)
    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="elu")(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = SpatialDropout2D(dropout_rate)(up6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = SpatialDropout2D(dropout_rate)(up7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = SpatialDropout2D(dropout_rate)(up8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = SpatialDropout2D(dropout_rate)(up9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv10 = Convolution2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)
    #model.summary()
    model.compile(optimizer=Adam(lr=learn_rate), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    
    #plot_model(model, to_file='model1.png',show_shapes=True)
    return model


def unet_fit(name, check_name = None):
    data_gen_args = dict(rotation_range=90.,   
                     width_shift_range=0.3,  
                     height_shift_range=0.3,   
                     horizontal_flip=True,   
                     vertical_flip=True,  
                     )
    from keras.preprocessing.image import ImageDataGenerator
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory(
        src,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator = mask_datagen.flow_from_directory(
        src,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1)


    image_datagen.fit(X_train)
    mask_generator.fit(X_mask_train)
    
    # combine generators into one which yields image and masks
    train_generator = itertools.izip(image_generator, mask_generator)   
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 40, 
                                   verbose = 1),
    ModelCheckpoint(model_paths + '{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    
    if check_name is not None:
        check_model = model_paths + '{}.h5'.format(check_name)
        model = load_model(check_model, 
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model(dropout_rate = 0.35, learn_rate = 1e-5, width = 32)
    model.fit_generator(
        train_generator,
        epochs=300,
        verbose =1, 
        callbacks = callbacks,
        steps_per_epoch=256,
        validation_data = train_generator,
        nb_val_samples = 48)
    return

all_new_imgs=[]
all_new_node_masks=[]
all_new_lung_masks=[]


history=unet_fit(name="unet_luna")
history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label="train loss")
plt.plot(epochs,val_loss_values,'b',label='validation loss')
plt.title("train and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
