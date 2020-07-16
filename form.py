import tkinter as tk
from tkinter import *
from tkinter import filedialog 
from PIL import ImageTk, Image
import os
import cv2
import numpy as np

class Application(tk.Frame):
    def __init__(self, master=None):       
        super().__init__(master)		# gọi hàm init cha
        self.listStep = []
        self.master.title("Xem quá trình")		#tên form
        self.master.minsize(1000,600)	#kích cỡ tối thiểu
        self.master = master
        self.pack()						#gọi hàm tạo widgets (vật dụng)
        self.create_canvas()


    def create_canvas(self):
        for i in range(8):
            step = Canvas(master = self, width = 250, height = 300, background = "Gray")     #thông số canvas 
            step.place(relx=(i%4)/4.0,rely=i/4.0)  
            step.pack()
            self.listStep.append(step)

root = tk.Tk()						#root là tên biến tự đặt, tk là thư viện tkinter, Tk() là hàm tạo form
app = Application(master=root)		
app.mainloop()						#chạy lặp form


