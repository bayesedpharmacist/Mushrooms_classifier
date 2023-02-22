#!/usr/bin/env python

# import os
from pandas import DataFrame
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from numpy import expand_dims, concatenate, argmax, argsort
import Shrooms_classifier.data.data_loader as dl

# path = dl.find_path_to_data(os.getcwd())


def find_improper_input(question, *possible_answers):

    """Takes the input of the user describing the visible features and checks for correctness of the input"""

    shroom_feature = input(question)
    while True:
        if shroom_feature in possible_answers:
            # print('This is a valid answer. Thank you!')
            break
        else:
            shroom_feature = input('This is not a valid answer. Please, try again.\n')
    return shroom_feature


hymenium_user = find_improper_input(
    'Please, tell me what is the shape of the hymenium of your mushroom?'
    ' You can choose from: gills, pores, or teeth: \n',
    'gills', 'pores', 'teeth')
hymenium_color_user = find_improper_input(
    'What is the color of the hymenium? '
    'Choose from: black, brown, gray, pink, pink-brown, pink-white, red, violet, white, yellow: \n',
    'black', 'brown', 'gray', 'pink', 'pink-brown', 'pink-white', 'red', 'violet', 'white', 'yellow')
ring_user = find_improper_input('Does this mushroom have a ring? Type: yes or no: \n', 'yes', 'no')
cap_color_user = find_improper_input(
    'What is the color of the cap of your mushroom? Choose from: black, blue, brown, gray, greenish, pink, red, '
    'violet, white, yellow: \n', 'black', 'blue', 'brown', 'gray', 'greenish', 'pink', 'red', 'violet', 'white', 'yellow')
scales_user = find_improper_input('Does this mushroom have scales on the cap? Type: yes or no: \n', 'yes', 'no')

img_path = input('Where is your image stored? \n')

user_values = (hymenium_user, hymenium_color_user, ring_user, cap_color_user, scales_user)


class create_user:

    """Combines all necessary transformed values of the user input and prepares them to give as input to the model"""

    def __init__(self, user_values, user_df_family, img_path):
        self.user_values = user_values
        self.user_df = DataFrame([user_values],
                                 columns=('hymenium', 'hymenium_color', 'ring', 'cap_color', 'cap_scales'))
        self.user_df_family = dl.user_df_family
        self.img_path = img_path

    def user_family(self):
        self.user_df_oe = dl.User_ordinal_encoder.transform(self.user_df)
        self.user_class = dl.User_cNB.classes_[argmax(dl.User_cNB.predict_proba(self.user_df_oe))]
        self.user_df_mapped = self.user_df_family.loc[self.user_df_family['family'] == self.user_class].drop(
            columns='family')
        return self.user_df_mapped

    def user_features(self):
        self.hymenium = dl.ohe_hymenium.transform(self.user_df[self.user_df.columns[0]].values.reshape(-1, 1))
        self.hymenium_color = dl.ohe_hymenium_color.transform(self.user_df[self.user_df.columns[1]].values.reshape(-1, 1))
        self.ring = dl.ohe_ring.transform(self.user_df[self.user_df.columns[2]].values.reshape(-1, 1))
        self.cap_color = dl.ohe_cap_color.transform(self.user_df[self.user_df.columns[3]].values.reshape(-1, 1))
        self.cap_scales = dl.ohe_cap_scales.transform(self.user_df[self.user_df.columns[4]].values.reshape(-1, 1))
        self.array = concatenate([self.hymenium, self.hymenium_color, self.ring, self.cap_color, self.cap_scales],
                                 axis=1)
        return self.array

    def preprocess_img(self):
        self.img = load_img(self.img_path, target_size=(256, 256))
        self.img_array = img_to_array(self.img)
        self.img_aug = dl.img_augment.standardize(self.img_array)
        self.test_img = expand_dims(self.img_aug, axis=0)
        return self.test_img

    def combine(self):
        self.user = (create_user.user_family(self), create_user.user_features(self), create_user.preprocess_img(self))
        return self.user


def predict_lookup(user):

    """Sorts the maximum probabilities and maps the corresponding species from a classes.df file."""

    predict_proba = model.predict(user).round(decimals=3)
    max_probability_idx = argmax(predict_proba)
    max_probability = predict_proba[0][max_probability_idx].round(decimals=3)
    max_class = classes_df.loc[max_probability_idx][0]
    first_three_idx = (-predict_proba).argsort()[:, 0: 3]
    second_class = classes_df.loc[first_three_idx[0][1]][0]
    second_class_probability = predict_proba[0][first_three_idx[0][1]].round(decimals=3)
    third_class = classes_df.loc[first_three_idx[0][2]][0]
    third_class_probability = predict_proba[0][first_three_idx[0][2]]
    return (max_class, max_probability), (second_class, second_class_probability), (third_class, third_class_probability)


classes_df = dl.classes_df
model = dl.model


user = create_user(user_values, dl.user_df_family, img_path)
user_to_predict = user.combine()
