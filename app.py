import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Load the trained model
model = load_model('models/emotion_classifier.h5')

# Define the class labels
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

# Create the main window
root = tk.Tk()
root.title('Emotion Classifier')
root.geometry('600x700')
root.resizable(False, False)
root.configure(background='#ffffff')

# Function to load and display the image
def upload_image():
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]
        )
        if file_path:
            # Load and preprocess the image
            img = Image.open(file_path).convert('RGB')
            img_resized = img.resize((48, 48))
            img_array = np.array(img_resized) / 255.0
            img_expanded = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_expanded)
            confidence = np.max(prediction)
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            
            # Update GUI elements
            display_image(img)
            result_label.config(
                text=f'Predicted Emotion: {predicted_class}\nConfidence: {confidence:.2f}'
            )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to display the image in the GUI
def display_image(img):
    img_display = ImageTk.PhotoImage(img.resize((250, 250)))
    image_label.configure(image=img_display)
    image_label.image = img_display

# Function to reset the application
def reset():
    image_label.configure(image='')
    image_label.image = None
    result_label.config(text='Predicted Emotion: None')

# Title label
title_label = tk.Label(
    root,
    text='Emotion Classifier',
    font=('Arial', 24, 'bold'),
    background='#ffffff',
    foreground='#333333'
)
title_label.pack(pady=20)

# Image display label
image_label = tk.Label(root, background='#ffffff')
image_label.pack(pady=10)

# Button frame
button_frame = tk.Frame(root, background='#ffffff')
button_frame.pack(pady=10)

# Upload button
upload_button = tk.Button(
    button_frame,
    text='Upload Image',
    command=upload_image,
    font=('Arial', 14),
    bg='#4CAF50',
    fg='white',
    activebackground='#45a049',
    cursor='hand2',
    width=15
)
upload_button.grid(row=0, column=0, padx=10)

# Reset button
reset_button = tk.Button(
    button_frame,
    text='Reset',
    command=reset,
    font=('Arial', 14),
    bg='#f44336',
    fg='white',
    activebackground='#e53935',
    cursor='hand2',
    width=15
)
reset_button.grid(row=0, column=1, padx=10)

# Result label
result_label = tk.Label(
    root,
    text='Predicted Emotion: None',
    font=('Arial', 18),
    background='#ffffff',
    foreground='#333333'
)
result_label.pack(pady=20)

# Start the GUI event loop
root.mainloop()
