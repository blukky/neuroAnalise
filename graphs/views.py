from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import tensorflow as tf
from keras.models import load_model
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import json

def get_data_from_file(df):
    df.columns = [i.lower() for i in df.columns]
    X1 = list()
    X2 = list()
    X3 = list()
    ID = list()
    for user, data_user in df.groupby("filename"):
        for test, test_data in data_user.groupby(["test_index"]):
            for repert, repert_data in test_data.groupby(["presentation"]):
                X_1 = list()
                X_2 = list()
                X_3 = list()
                for ind, question in enumerate(
                        list(repert_data.question)):
                    data = repert_data[repert_data["question"] == question]
                    data_1 = np.array([int(i) * (-1) for i in list(data.data)[0][1:-1].split(", ")])
                    data_2 = np.array([int(i) * (-1) for i in list(data.data_2)[0][1:-1].split(", ")])
                    signals, info = nk.ppg_process(data_1, sampling_rate=20)
                    X_1.append(data_1)
                    X_2.append(data_2)
                    X_3.append(np.array(list(signals.PPG_Rate)))
                    ID.append([user, str(test), str(repert), str(question)])
                if len(X_1) == 0:
                    continue
                X_1 = np.array(X_1).reshape((len(X_1) * 240,))
                X_2 = np.array(X_1).reshape((len(X_2) * 240,))
                X_3 = np.array(X_2).reshape((len(X_3) * 240,))
                X_1 = (X_1 - X_1.min()) / (X_1.max() - X_1.min())
                X_3 = (X_3 - X_3.min()) / (X_3.max() - X_3.min())
                X_2 = (X_2 - X_2.min()) / (X_2.max() - X_2.min())
                X1 += [X_1[i * 240:240 * (i + 1)] for i in range(X_1.shape[0] // 240)]
                X3 += [X_3[i * 240:240 * (i + 1)] for i in range(X_3.shape[0] // 240)]
                X2 += [X_2[i * 240:240 * (i + 1)] for i in range(X_2.shape[0] // 240)]
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    return X1, X2, X3, ID


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 240
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [240] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=130, frame_step=1)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


model_nn = load_model("first_model_nn.h5")
model_cnn = load_model("first_model_cnn.h5")


def index(request):
    return render(request, 'index.html', )


def api(request):
    return render(request, 'api.html')


def convert_spectogramm(X1, X2, X3):
    X1S = list()
    X2S = list()
    X3S = list()
    for i in range(len(X1)):
        X1S.append(get_spectrogram(X1[i]))
        X2S.append(get_spectrogram(X2[i]))
        X3S.append(get_spectrogram(X3[i]))
    X1S = np.array(X1S, dtype=np.float64)
    X2S = np.array(X2S, dtype=np.float64)
    X3S = np.array(X3S, dtype=np.float64)
    return X1S, X2S, X3S


def predict(df):
    X1, X2, X3, ID = get_data_from_file(df)
    X1S, X2S, X3S = convert_spectogramm(X1, X2, X3)
    result = [str(np.argmax(i)) for i in model_nn.predict((X1, X2, X3))]
    return result, ID, X1, X2, X3


def analysis(request):
    if request.FILES:
        file = request.FILES.get("file")
        if file.name.split(".")[-1] != "csv":
            return JsonResponse(status=400, data={"detail": "Неверный формат файла"})
        with open(f'data.csv', 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        # try:
        df = pd.read_csv("data.csv")
        result, id, X1, X2, X3 = predict(df)
        os.remove("data.csv")
        arr = json.dumps({"result": result, "id": id, "x1": X1.tolist(), "x2": X2.tolist(), "x3": X3.tolist()})
        return JsonResponse(status=200, data={"data": arr})

    data_string = request.POST.get('data_string')
    data = np.array([int(i) * (-1) for i in data_string[1:-1].split(", ")])
    if len(data) != 240:
        return JsonResponse(status=400, data={"detail": "Неверный формат файла"})
    data = (data - data.min()) / (data.max() - data.min())
    data = np.expand_dims(data, axis=0)
    result = np.argmax(model_nn.predict(data)[0])
    return JsonResponse(status=200, data={"class": result})
