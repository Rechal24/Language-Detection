from PIL.ImageTk import PhotoImage
from ultralytics import YOLO
import os, shutil
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

#initialize GUI
np.set_printoptions(suppress=True)
top = tk.Tk()
top.geometry('1000x600') #adjust width for better layout
top.title("Language Detection")
#img = PhotoImage(file='.png', master=top)
#img_label = Label(top, image=img)
#img_label.place(x=0, y=0)

uploaded_file_path = ""

#function to classify image and display confidence score and class name
def classify():
    global uploaded_file_path

    if uploaded_file_path == "":
        confidence_label.configure(text='no image uploaded', fg='red')
        class_label.configure(text='no prediction available', fg='red')
        return

    path = "output"
    if os.path.exists(path):
        #if directory exists, delete it
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    model = YOLO("best.pt")

    #update class names
    model.names[0] = "english"
    model.names[1] = "german"
    model.names[2] = "kannada"

    results = model.predict(source=uploaded_file_path, project="output", save=True, save_txt=True, conf=0.1)

    #extract the top prediction and confidence score
    result = results[0]
    boxes = result.boxes #yolov8 returns bboxes and their confidences

    if len(boxes) > 0:
        #get the highest confidence prediction
        top_index = np.argmax(boxes.conf.numpy())#get the index of highest confidence
        top.confidence = boxes.conf[top_index].item() #get the confidence score
        predicted_class = model.names[int(boxes.cls[top_index].item())] #get the class name

        #display result image
        filename = os.path.splitext(os.path.basename(uploaded_file_path))[0]
        print("predicted file path:" + "output/predict/" + filename + ".jpg")
        im = Image.open(r"output/predict/" + filename + ".jpg")
        im.save("output/predict/predictedimage.jpg")
        uploaded = Image.open("output/predict/predictedimage.jpg")
        uploaded.thumbnail((top.winfo_width() // 2, top.winfo_height() // 2))
        im = ImageTk.PhotoImage(uploaded)
        resultimg.configure(image=im)
        resultimg.image = im

        #display confidence score and predicted class
        confidence_label.configure(text=f'confidence: {top.confidence:.2f}', fg='green')
        class_label.configure(text=f'predicted class: {predicted_class}', fg='green')

    else:
        confidence_label.configure(text='no detection', fg='red')
        class_label.configure(text='no prediction available', fg='red')

#upload image function
def upload_image():
    global uploaded_file_path
    try:
        uploaded_file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", ".jpg;.jpeg;.png"), ("JPEG files",".jpg;.jpeg"), ("PNG files", ".png")])#store the file path
        if uploaded_file_path: #if a file was selected
            uploaded = Image.open(uploaded_file_path)
            uploaded.thumbnail(((top.winfo_width() / 3), (top.winfo_height() / 3))) #resize image for
            im = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image = im
            label.configure(text='image uploaded! now click "classify image" to classify it.', fg='black')
            confidence_label.configure(text='Confidence: N/A') #reset confidence label
            class_label.configure(text='predicted class: N/A')#reset class label
        else:
            label.configure(text='no image selected', fg='red')
    except Exception as e:
        print(f"Error: {e}")


#instruction label
label = Label(top, font=('Helvetica', 12), bg='#F0F0F0', fg='#555555')
label.grid(row=6, column=0, columnspan=2, pady=20)

#label for uploaded image
sign_image = Label(top, bg='#F0F0F0')
sign_image.grid(row=3, column=0, padx=30, pady=20)

#label for result image
resultimg = Label(top, bg='#F0F0F0')
resultimg.grid(row=3, column=1, padx=30, pady=20)

#confidence score label
confidence_label = Label(top, text="Confidence:N/A", font=('Helvetica', 14), bg='#F0F0F0', fg='#000000')
confidence_label.grid(row=4, column=0, columnspan=2, pady=10)

#predicted class label
class_label = Label(top, text="predicted Class: N/A", font=('Helvetica', 14), bg='#F0F0F0', fg='#000000')
class_label.grid(row=4)

#upload  image button
upload = Button(top, text="upload image", command=upload_image, padx=15, pady=10)
upload.configure(background='#5E81AC', foreground='white', font=('Helvetica', 14, 'bold'), relief="flat", borderwidth=0)
upload.grid(row=7, column=0, padx=20, pady=10)

#classify image label
classify_image =Button(top, text="classify image", command=classify, padx=15, pady=10)
classify_image.configure(background='#5E81AC', foreground='white', font=('Helvetica', 14, 'bold'), relief="flat", borderwidth=0)
classify_image.grid(row=7, column=1, padx=20, pady=10)

#heading with clean, large font
heading = Label(top, text="Language Detection", pady=20, font=('Helvetica', 14, 'bold'))
heading.configure(background='#F0F0F0', foreground='#2E3440')
heading.grid(row=0, column=0, columnspan=2)

#center the layout
top.grid_columnconfigure(0, weight=1)
top.grid_columnconfigure(1, weight=1)
top.grid_rowconfigure(0, weight=1)
top.grid_rowconfigure(1, weight=1)
top.grid_rowconfigure(2, weight=1)
top.grid_rowconfigure(3, weight=1)
top.grid_rowconfigure(4, weight=1)
top.grid_rowconfigure(5, weight=1)
top.grid_rowconfigure(6, weight=1)
top.grid_rowconfigure(7, weight=1)

top.mainloop()