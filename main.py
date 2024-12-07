import numpy as np
import pandas as pd
import pickle as pkl
import tkinter as tk

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tkinter import Button, ttk
from tkinter import messagebox

df = pd.read_csv('crime_prediction_in_chicago_dataset.csv',
                 usecols=['Primary Type', 'Location Description']).dropna()

MODEL_TYPE_VALUES = ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Support Vector Machine']

PRIMARY_TYPE_VALUES = df['Primary Type'].unique().tolist()

LOCATION_DESCRIPTION_VALUES = df['Location Description'].unique().tolist()

with open('./saved_models/decision_tree_classifier.pkl', 'rb') as file:
  decision_tree_classifier = pkl.load(file)

with open('./saved_models/k_neighbors_classifier.pkl', 'rb') as file:
  k_neighbors_classifier = pkl.load(file)

with open('./saved_models/logistic_regression.pkl', 'rb') as file:
  logistic_regression = pkl.load(file)

with open('./saved_models/svm_classifier.pkl', 'rb') as file:
  svm_classifier = pkl.load(file)

with open('./saved_models/encoder.pkl', 'rb') as file:
  encoder = pkl.load(file)

def validate_input(primary_type, location_description, community_area, model_type):
  if primary_type not in PRIMARY_TYPE_VALUES:
    messagebox.showerror('Invalid Input!', 'Please select a valid Primary Type.')

    return False
  elif location_description not in LOCATION_DESCRIPTION_VALUES:
    messagebox.showerror('Invalid Input!', 'Please select a valid Location Description.')

    return False
  elif model_type not in MODEL_TYPE_VALUES:
    messagebox.showerror('Invalid Input!', 'Please select a valid Model Type.')

    return False
  else:
    try:
      community_area = int(community_area)

      if community_area < 1 or community_area > 77:
        messagebox.showerror('Invalid Input!', 'Please select a valid Community Area (from 1 to 77).')

        return False
    except ValueError:
      messagebox.showerror('Invalid Input!', 'Please select a valid Community Area (from 1 to 77).')

      return False

  return True

def show_prediction():
  primary_type = primary_type_box.get()
  location_description = location_description_box.get()
  community_area = community_area_box.get()
  model_type = model_type_box.get()

  if not validate_input(primary_type, location_description, community_area, model_type): return

  prediction_input = [primary_type, location_description, community_area]

  match model_type:
    case 'decision tree':
      prediction_text.insert(tk.END, f'{decision_tree_classifier.predict(prediction_input)}\n')
      prediction_text.insert(tk.END, f'{decision_tree_classifier.predict_prob(prediction_input)}')
      prediction_text.insert(tk.END, 'decision tree\n')
    case 'k-neighbors':
      prediction_text.insert(tk.END, f'{k_neighbors_classifier.predict(prediction_input)}\n')
      prediction_text.insert(tk.END, f'{k_neighbors_classifier.predict_prob(prediction_input)}')
    case 'logistic regression':
      prediction_text.insert(tk.END, f'{logistic_regression.predict(prediction_input)}')
      prediction_text.insert(tk.END, f'{logistic_regression.predict_prob(prediction_input)}')
    case 'svm':
      prediction_text.insert(tk.END, f'{svm_classifier.predict(prediction_input)}\n')
      prediction_text.insert(tk.END, f'{svm_classifier.predict_prob(prediction_input)}')

root = tk.Tk()
root.geometry('600x600')
root.title('Crime Rate in Chicago')

primary_type_frame = tk.Frame(root)
primary_type_frame.pack(pady=10, padx=10, fill='x')

primary_type_label = tk.Label(primary_type_frame, text='Primary Type:', font=('Arial', 14))
primary_type_label.pack(side=tk.LEFT, pady=(10, 5), anchor='w')

primary_type_box = ttk.Combobox(primary_type_frame, values=PRIMARY_TYPE_VALUES, font=('Arial', 12), width=30)
primary_type_box.pack(side=tk.LEFT, pady=5, padx=10)

location_description_frame = tk.Frame(root)
location_description_frame.pack(pady=10, padx=10, fill='x')

location_description_label = tk.Label(location_description_frame, text='Location Description:', font=('Arial', 14))
location_description_label.pack(side=tk.LEFT, pady=(10, 5), anchor='w')

location_description_box = ttk.Combobox(location_description_frame, values=LOCATION_DESCRIPTION_VALUES, font=('Arial', 12), width=40)
location_description_box.pack(side=tk.LEFT, pady=5, padx=10)

community_area_frame = tk.Frame(root)
community_area_frame.pack(pady=10, padx=10, fill='x')

community_area_label = tk.Label(community_area_frame, text='Community Area:', font=('Arial', 14))
community_area_label.pack(side=tk.LEFT, padx=(0, 2))

community_area_box = tk.Spinbox(community_area_frame, from_=1, to=77, font=('Arial', 12))
community_area_box.pack(side=tk.LEFT, pady=10)

model_type_frame = tk.Frame(root)
model_type_frame.pack(pady=10, padx=10, fill='x')

model_label = tk.Label(model_type_frame, text='Model:', font=('Arial', 14))
model_label.pack(side=tk.LEFT, pady=(10, 5), anchor='w')

model_type_box = ttk.Combobox(model_type_frame, values=MODEL_TYPE_VALUES, font=('Arial', 12), width=20)
model_type_box.pack(side=tk.LEFT, pady=5, padx=10)

button = Button(root, text='OK', font=('Arial', 12), command=show_prediction, width=8)
button.pack(pady=5, padx=10)

prediction_frame = tk.Frame(root)
prediction_frame.pack(padx=10, pady=10)

prediction_text = tk.Text(prediction_frame, height=50, width=50)
prediction_text.pack(side=tk.LEFT)

scrollbar = tk.Scrollbar(prediction_frame, orient=tk.VERTICAL, command=prediction_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

prediction_text.config(yscrollcommand=scrollbar.set)

root.mainloop()
