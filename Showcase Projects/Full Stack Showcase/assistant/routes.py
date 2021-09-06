from assistant import app
from flask import render_template, redirect, url_for, flash, request
from assistant.models import User
from assistant.forms import RegisterForm, LoginForm
from assistant import db
from flask_login import login_user, logout_user, login_required

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import speech_recognition as sr
import playsound
# import pyttsx3 as tts
# import sys
import time
import os
from gtts import gTTS


r = sr.Recognizer()

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('assistant/intents.json').read())

words = pickle.load(open('assistant/words.pkl', 'rb'))
classes = pickle.load(open('assistant/classes.pkl', 'rb'))
model = load_model('assistant/chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, q] for i, q in enumerate(res) if q > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for q in results:
        return_list.append({'intent': classes[q[0]], 'probability': str(q[1])})
    return return_list


def record_audio():
    with sr.Microphone() as source:
        audio = r.listen(source)
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)
        except sr.UnknownValueError:
            speak('did not understand')
        except sr.RequestError:
            speak('service is down')
        return voice_data


def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en')
    a = random.randint(1, 10000000)
    audio_file = 'audio-' + str(a) + '.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/assistant', methods=['GET', 'POST'])
@login_required
def assistant_page():
    return render_template('assistant.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email_address=form.email_address.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully! You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('assistant_page'))
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'There was an error with creating a user: {err_msg}', category='danger')

    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('assistant_page'))
        else:
            flash('Username and password are not match! Please try again', category='danger')

    return render_template('login.html', form=form)


@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))


@app.route('/voice')
def testing():
    print('In SomeFunction')
    time.sleep(1)
    while 1:
        voice_data = record_audio()
        message = voice_data
        print(message)
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
        print('break')
        speak(res)

