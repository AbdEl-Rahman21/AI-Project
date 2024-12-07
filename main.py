import tkinter as tk
from cProfile import label
from tkinter import ttk, Button
import pickle
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

with open('decision_tree_classifier.pkl', 'rb') as f:
    decision_tree_classifier = pickle.load(f)

with open('k_neighbors_classifier.pkl', 'rb') as f:
        k_neighbors_classifier = pickle.load(f)

with open("logistic_regression.pkl", 'rb') as f:
        logistic_regression = pickle.load(f)

with open("svm_classifier.pkl", 'rb') as f:
    svm_classifier = pickle.load(f)


def validate_number(min, max):
    try:

        number = int(spinbox.get())

        if number < min or number > max:
            messagebox.showerror("Invalid Input", "Please enter a number between 1 and 10.")
            spinbox.delete(0, tk.END)
            spinbox.insert(0, "0")

    except ValueError:

        messagebox.showerror("Invalid Input", "Please enter a valid number.")
        spinbox.delete(0, tk.END)  # Clear the invalid input
        spinbox.insert(0, "0")

def show_model():
    validate_number(0, 77)
    primaryType = PrimaryType_box.get()
    locationDescription = LocationDescription_box.get()
    spinbox_value = int(spinbox.get())
    modelType = Model_box.get()

    prediction_input = [[primaryType, float(spinbox_value) ,locationDescription]]

    match modelType:
        case "decision tree":
            #t.insert(tk.END, f"{decision_tree_classifier.predict(prediction_input)}\n")
            #t.insert(tk.END,f"{decision_tree_classifier.predict_prob(prediction_input)}")
            t.insert(tk.END, "decision tree\n")
        case "k-neighbors":
            t.insert(tk.END,f"{k_neighbors_classifier.predict(prediction_input)}\n")
            t.insert(tk.END, f"{k_neighbors_classifier.predict_prob(prediction_input)}")
        case "logistic regression":
            t.insert(tk.END,f"{logistic_regression.predict(prediction_input)}")
            t.insert(tk.END, f"{logistic_regression.predict_prob(prediction_input)}")
        case "svm":
            t.insert(tk.END, f"{svm_classifier.predict(prediction_input)}\n")
            t.insert(tk.END, f"{svm_classifier.predict_prob(prediction_input)}")


root = tk.Tk()
root.title("Crime Rate in Chicago")
root.geometry("600x600")

p_typeFrame= tk.Frame(root)
p_typeFrame.pack(pady=10, padx=10, fill="x")

PrimaryType_label = tk.Label(p_typeFrame, text="Primary Type:", font=("Arial", 14))
PrimaryType_label.pack(side = tk.LEFT,pady=(10, 5), anchor="w")

PrimaryType_box = ttk.Combobox(p_typeFrame, values=["Option 1", "Option 2", "Option 3"], font=("Arial", 12), width= 20)
PrimaryType_box.pack(side = tk.LEFT, pady=5, padx=10)


description_Frame= tk.Frame(root)
description_Frame.pack(pady=10, padx=10, fill="x")

LocationDescription_label = tk.Label(description_Frame, text="Location Description:", font=("Arial", 14))
LocationDescription_label.pack(side = tk.LEFT, pady=(10, 5), anchor="w")

LocationDescription_box = ttk.Combobox(description_Frame, values=["Option 1", "Option 2", "Option 3"], font=("Arial", 12), width= 20)
LocationDescription_box.pack(side = tk.LEFT, pady=5, padx=10)

spin_frame = tk.Frame(root)
spin_frame.pack(pady=10, padx=10, fill="x")

label_spinbox = tk.Label(spin_frame, text="Enter Number: ", font=("Arial", 14))
label_spinbox.pack(side = tk.LEFT ,padx=(0, 2))

spinbox = tk.Spinbox(spin_frame, from_=0, to=77, font=("Arial", 12), validate="key")
spinbox.pack(side = tk.LEFT ,pady=10)

model_frame = tk.Frame(root)
model_frame.pack(pady=10, padx=10, fill="x")
Model_label = tk.Label(model_frame, text="Model:", font=("Arial", 14))
Model_label.pack(side = tk.LEFT, pady=(10, 5), anchor="w")

Model_box = ttk.Combobox(model_frame, values=["decision tree", "k-neighbors", "logistic regression", "svm"], font=("Arial", 12) , width= 20)
Model_box.pack(side = tk.LEFT, pady=5, padx=10)

show_info = Button(root, text="OK", font=("Arial", 12), command=show_model, width= 8)
show_info.pack(pady=5, padx=10)



frame = tk.Frame(root)
frame.pack(padx=10, pady=10)


t = tk.Text(frame, height=50, width=50)
t.pack(side=tk.LEFT)


scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=t.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


t.config(yscrollcommand=scrollbar.set)


root.mainloop()