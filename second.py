import socket
import base64
import os
import select
import glob
import numpy as np
import tensorflow as tf
import librosa
import sys
import time
import csv
from sklearn.model_selection import train_test_split

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

host = ""
port = 8889

# Setup client socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 8888))
s.send("second".encode())
parse = s.recv(1024).decode()

if parse == "yes" or "no":
	print("Authorisation Success.")
	print("Second Ready.")
	while True:
		data = s.recv(1024).decode()
		form = "second"
		print("Form:", form)
		# print("Data:", data)
		s.send(form.encode())
		# s.send(data.encode())
		start = time.time()
		print(" ")
		print("==================")
		print(" REQUEST RECEIVED ")
		print("==================")
		print(" ")
		print("Data:",str(data))
		data2 = s.recv(1024).decode() # data2 is the series info
		sound_info = s.recv(1024).decode() 
		image_info = s.recv(1024).decode()
		print("Data2:",str(data2))
		print("image_info: ",str(image_info))
		P1,P2,P3,P4,P5,P6 = data2.split(" ") # Plastic, Metal, Glass
		SP1,SP2,SP3 = sound_info.split(" ") # Plastic, Metal, Glass
		IP1,IP2,IP3 = image_info.split(" ") # Plastic, Metal, Glass
		print("Sound:",str(SP1)," Metal:",str(SP2), " Glass:",str(SP3))
		# print("Adding SP1, SP2 and SP3 =", float(SP1)+float(SP2)+float(SP3))
		if (float(P2) == 1.0) & (float(P5) >= 0.5): #& (float(SP2) >= 0.5):
			label = 'Metal'
			print(label) # Metal
			for i in range(3):
				if i == 0:
					source_filename = "predict/image.jpg"
					form = "Image" # Image
					format = "jpg"
				if i == 1:
					source_filename = "predict/sound.wav"
					form = "Sound" # Sound
					format = "wav"
				if i == 2:
					source_filename = "predict/series.csv"
					form = "Series" # Series
					format = "csv"
				filepath = "Learned/" + format + "/" + label
				count = str(len([f for f in os.listdir(filepath)]))
				filename = filepath + "/" + count + "." + format
				with open(source_filename, "rb") as f:
					data3 = f.read()
				with open(filename, "wb") as f:
					# print("filename: ", filename)
					f.write(data3) # base64.b64decode(data3)
			existing_features = np.load('processed/sound_features.npy')
			existing_labels = np.load('processed/sound_labels.npy')
			new_features = []
			new_labels = []
			for wav_name in glob.glob("predict/sound.wav"):
				mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)	
				extra_features = []
				extra_features.extend(mfccs)
				extra_features.extend(chroma)
				extra_features.extend(mel)
				extra_features.extend(contrast)
				extra_features.extend(tonnetz)
				new_features.append(extra_features)
				new_labels.append(1) # Metal
			new_features = np.array(new_features)
			new_labels = np.array(new_labels)
			new_features = np.concatenate((existing_features, new_features))
			new_labels = np.concatenate((existing_labels, new_labels))
			np.save('processed/sound_features.npy', new_features)
			np.save('processed/sound_labels.npy', new_labels)

			existing_freqs = np.load('processed/series_freqs.npy')
			existing_masses = np.load('processed/series_masses.npy')
			existing_series_labels = np.load('processed/series_labels.npy')
			new_series_freqs = []
			new_series_masses = []
			new_series_labels = []
			freqs = []
			masses = []
			labels = []
			for csv_name in glob.glob("predict/series.csv"):
				with open(csv_name, "r") as f:
					reader = csv.DictReader(f)
					freq_list = []
					mass_list = []
					for row in reader:
						freq = float(row['Freq'])
						freq_list.append(freq)
						mass = float(row['Mass'])
						mass_list.append(mass)
					freqs.append(freq_list)
					masses.append(mass_list)
					labels.append(1) # Metal
			new_series_freqs = np.array(freqs)
			new_series_masses = np.array(masses)
			new_series_labels = np.array(labels)
			new_series_freqs = np.concatenate((existing_freqs, new_series_freqs))
			new_series_masses = np.concatenate((existing_masses, new_series_masses))
			new_series_labels = np.concatenate((existing_labels, new_series_labels))
			np.save('processed/series_freqs.npy', new_series_freqs)
			np.save('processed/series_masses.npy', new_series_masses)
			np.save('processed/series_labels.npy', new_series_labels)

			existing_combined_features = np.load('processed/combined_features.npy')
			existing_combined_labels = np.load('processed/combined_labels.npy')
			combined_features = []
			combined_labels = []
			label_encoding = {"Plastic": 0, "Metal": 1, "Glass": 2}
			image_features = image_info
			sound_features = sound_info
			series_features = data2
			image_features = image_features.split(" ")
			sound_features = sound_features.split(" ")
			series_features = series_features.split(" ")
			feature_vector = []
			feature_vector.extend(image_features)
			feature_vector.extend(sound_features)
			feature_vector.extend(series_features)
			feature_vector = list(map(float,feature_vector))
			combined_features.append(feature_vector)
			combined_labels.append(1)
			test_features = np.array(combined_features)
			test_labels = np.array(combined_labels)
			np.save('processed/test_combined_labels.npy', test_labels)
			np.save('processed/test_combined_features.npy', test_features)
			print("Features:", test_features.shape)
			print("Labels:", test_labels.shape)

			combined_features = np.concatenate((existing_combined_features, test_features))
			combined_labels = np.concatenate((existing_combined_labels, test_labels))
			# SAVE
			np.save('processed/combined_features.npy', combined_features)
			np.save('processed/combined_labels.npy', combined_labels)


		elif (float(P6) == 1.0) & (float(P2) == 0.0): #& (float(SP3) >= 0.5):
			label = 'Glass'
			print(label) # Glass
			for i in range(3):
				if i == 0:
					source_filename = "predict/image.jpg"
					form = "Image" # Image
					format = "jpg"
				if i == 1:
					source_filename = "predict/sound.wav"
					form = "Sound" # Sound
					format = "wav"
				if i == 2:
					source_filename = "predict/series.csv"
					form = "Series" # Series
					format = "csv"
				filepath = "Learned/" + format + "/" + label
				count = str(len([f for f in os.listdir(filepath)]))
				filename = filepath + "/" + count + "." + format
				with open(source_filename, "rb") as f:
					data3 = f.read()
				with open(filename, "wb") as f:
					# print("filename: ", filename)
					f.write(data3) # base64.b64decode(data3)
			existing_features = np.load('processed/sound_features.npy')
			existing_labels = np.load('processed/sound_labels.npy')
			new_features = []
			new_labels = []
			for wav_name in glob.glob("predict/sound.wav"):
				mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)	
				extra_features = []
				extra_features.extend(mfccs)
				extra_features.extend(chroma)
				extra_features.extend(mel)
				extra_features.extend(contrast)
				extra_features.extend(tonnetz)
				new_features.append(extra_features)
				new_labels.append(2) # Glass
			new_features = np.array(new_features)
			new_labels = np.array(new_labels)
			new_features = np.concatenate((existing_features, new_features))
			new_labels = np.concatenate((existing_labels, new_labels))
			np.save('processed/sound_features.npy', new_features)
			np.save('processed/sound_labels.npy', new_labels)

			existing_freqs = np.load('processed/series_freqs.npy')
			existing_masses = np.load('processed/series_masses.npy')
			existing_series_labels = np.load('processed/series_labels.npy')
			new_series_freqs = []
			new_series_masses = []
			new_series_labels = []
			freqs = []
			masses = []
			labels = []
			for csv_name in glob.glob("predict/series.csv"):
				with open(csv_name, "r") as f:
					reader = csv.DictReader(f)
					freq_list = []
					mass_list = []
					for row in reader:
						freq = float(row['Freq'])
						freq_list.append(freq)
						mass = float(row['Mass'])
						mass_list.append(mass)
					freqs.append(freq_list)
					masses.append(mass_list)
					labels.append(2) # Glass
			new_series_freqs = np.array(freqs)
			new_series_masses = np.array(masses)
			new_series_labels = np.array(labels)
			new_series_freqs = np.concatenate((existing_freqs, new_series_freqs))
			new_series_masses = np.concatenate((existing_masses, new_series_masses))
			new_series_labels = np.concatenate((existing_labels, new_series_labels))
			np.save('processed/series_freqs.npy', new_series_freqs)
			np.save('processed/series_masses.npy', new_series_masses)
			np.save('processed/series_labels.npy', new_series_labels)

			existing_combined_features = np.load('processed/combined_features.npy')
			existing_combined_labels = np.load('processed/combined_labels.npy')
			combined_features = []
			combined_labels = []
			label_encoding = {"Plastic": 0, "Metal": 1, "Glass": 2}
			image_features = image_info
			sound_features = sound_info
			series_features = data2
			image_features = image_features.split(" ")
			sound_features = sound_features.split(" ")
			series_features = series_features.split(" ")
			feature_vector = []
			feature_vector.extend(image_features)
			feature_vector.extend(sound_features)
			feature_vector.extend(series_features)
			feature_vector = list(map(float,feature_vector))
			combined_features.append(feature_vector)
			combined_labels.append(2)
			test_features = np.array(combined_features)
			test_labels = np.array(combined_labels)
			print("Features:", test_features.shape)
			print("Labels:", test_labels.shape)

			combined_features = np.concatenate((existing_combined_features, test_features))
			combined_labels = np.concatenate((existing_combined_labels, test_labels))
			# SAVE
			np.save('processed/combined_features.npy', combined_features)
			np.save('processed/combined_labels.npy', combined_labels)

		elif (float(P6) == 0.0) & (float(P2) == 0.0) & (float(SP1) >= 0.99):
			label = 'Plastic'
			print(label) # Plastic
			for i in range(3):
				if i == 0:
					source_filename = "predict/image.jpg"
					form = "Image" # Image
					format = "jpg"
				if i == 1:
					source_filename = "predict/sound.wav"
					form = "Sound" # Sound
					format = "wav"
				if i == 2:
					source_filename = "predict/series.csv"
					form = "Series" # Series
					format = "csv"
				filepath = "Learned/" + format + "/" + label
				count = str(len([f for f in os.listdir(filepath)]))
				filename = filepath + "/" + count + "." + format
				with open(source_filename, "rb") as f:
					data3 = f.read()
				with open(filename, "wb") as f:
					# print("filename: ", filename)
					f.write(data3) # base64.b64decode(data3)
			existing_features = np.load('processed/sound_features.npy')
			existing_labels = np.load('processed/sound_labels.npy')
			new_features = []
			new_labels = []
			for wav_name in glob.glob("predict/sound.wav"):
				mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)	
				extra_features = []
				extra_features.extend(mfccs)
				extra_features.extend(chroma)
				extra_features.extend(mel)
				extra_features.extend(contrast)
				extra_features.extend(tonnetz)
				new_features.append(extra_features)
				new_labels.append(0) # Plastic
			new_features = np.array(new_features)
			new_labels = np.array(new_labels)
			new_features = np.concatenate((existing_features, new_features))
			new_labels = np.concatenate((existing_labels, new_labels))
			np.save('processed/sound_features.npy', new_features)
			np.save('processed/sound_labels.npy', new_labels)

			existing_freqs = np.load('processed/series_freqs.npy')
			existing_masses = np.load('processed/series_masses.npy')
			existing_series_labels = np.load('processed/series_labels.npy')
			new_series_freqs = []
			new_series_masses = []
			new_series_labels = []
			freqs = []
			masses = []
			labels = []
			for csv_name in glob.glob("predict/series.csv"):
				with open(csv_name, "r") as f:
					reader = csv.DictReader(f)
					freq_list = []
					mass_list = []
					for row in reader:
						freq = float(row['Freq'])
						freq_list.append(freq)
						mass = float(row['Mass'])
						mass_list.append(mass)
					freqs.append(freq_list)
					masses.append(mass_list)
					labels.append(0) # Plastic
			new_series_freqs = np.array(freqs)
			new_series_masses = np.array(masses)
			new_series_labels = np.array(labels)
			new_series_freqs = np.concatenate((existing_freqs, new_series_freqs))
			new_series_masses = np.concatenate((existing_masses, new_series_masses))
			new_series_labels = np.concatenate((existing_labels, new_series_labels))
			np.save('processed/series_freqs.npy', new_series_freqs)
			np.save('processed/series_masses.npy', new_series_masses)
			np.save('processed/series_labels.npy', new_series_labels)

			existing_combined_features = np.load('processed/combined_features.npy')
			existing_combined_labels = np.load('processed/combined_labels.npy')
			combined_features = []
			combined_labels = []
			label_encoding = {"Plastic": 0, "Metal": 1, "Glass": 2}
			image_features = image_info
			sound_features = sound_info
			series_features = data2
			image_features = image_features.split(" ")
			sound_features = sound_features.split(" ")
			series_features = series_features.split(" ")
			feature_vector = []
			feature_vector.extend(image_features)
			feature_vector.extend(sound_features)
			feature_vector.extend(series_features)
			feature_vector = list(map(float,feature_vector))
			combined_features.append(feature_vector)
			combined_labels.append(0) # Plastic
			test_features = np.array(combined_features)
			test_labels = np.array(combined_labels)
			print("Features:", test_features.shape)
			print("Labels:", test_labels.shape)

			combined_features = np.concatenate((existing_combined_features, test_features))
			combined_labels = np.concatenate((existing_combined_labels, test_labels))
			# SAVE
			np.save('processed/combined_features.npy', combined_features)
			np.save('processed/combined_labels.npy', combined_labels)

		


