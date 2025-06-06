import tkinter as tk
from tkinter import filedialog
import os
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model

def upload_image():
    file_path = filedialog.askopenfilename(initialdir="dataset/oldTest")
    image = Image.open(file_path)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    image_tk = ImageTk.PhotoImage(Image.open(file_path).resize((350, 350)))
    image_label.config(image=image_tk)
    image_label.image = image_tk

    result_label.config(text='')

    global uploaded_image
    uploaded_image = image

def predict_image():
    result = model.predict(uploaded_image)
    if result < 0.5:
        result_label.config(text='猫',font=("黑体", 36, "bold"))
        print("猫")
        result_label.pack(pady=20)
    else:
        result_label.config(text='狗',font=("黑体", 36, "bold"))
        result_label.pack(pady=20)
        print("狗")

# 加载模型
model = load_model('cat_dog_classify_model.keras')

# 创建GUI界面
base = tk.Tk()
base.title('猫狗识别系统')
base.geometry('900x700')  # 调整窗口大小

# 标题
title = tk.Label(base, text='猫狗识别系统',font=("黑体", 48, "bold"),fg='blue')
title.pack(pady=20)

# 菜单
menu = tk.Menu(base)
file_menu = tk.Menu(menu, tearoff=0)
file_menu.add_command(label="上传图片",command=upload_image)
file_menu.add_separator()
file_menu.add_command(label="识别",command=predict_image)
menu.add_cascade(label="操作", menu=file_menu, underline=0)
base.config(menu=menu)

image_frame = tk.Frame(base)
image_frame.pack()
image_label = tk.Label(image_frame)
image_label.pack(pady=20)

result_label = tk.Label(base, text='')
result_label.pack()

base.mainloop()


