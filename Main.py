# Main.py
import tkinter as tk
from tkinter import *
from tkinter import filedialog 
from PIL import ImageTk, Image

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

training = DetectChars.trainingData()
if training == False:                             
    print("\nerror: KNN traning was not successful\n")  
    exit()  

def main(path):

    imgOriginalScene  = cv2.imread(path)

    if imgOriginalScene is None:                     
        print("\nerror: image not read from file \n\n")  
        os.system("pause")                                  
        return     

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)       

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)       

    if len(listOfPossiblePlates) == 0:                    
        print("\nno license plates were detected\n")
        app.insertText('') 
    else: 
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]

        if DetectPlates.showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)       
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                 
            print("\nno characters were detected\n\n")  
            return   

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)            

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)        

        if DetectPlates.showSteps == True:
            cv2.imshow("imgOriginalScene", imgOriginalScene)             

        pilImg = cv2.cvtColor(imgOriginalScene, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(pilImg)
        app.changeImage(pilImage)
        app.insertText(licPlate.strChars)
    return

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)     

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)      
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                          
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                      
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                    
    fltFontScale = float(plateHeight) / 30.0                  
    intFontThickness = int(round(fltFontScale * 1.5))         

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)      

    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)           
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)       

    if intPlateCenterY < (sceneHeight * 0.75):                                                
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))    
    else:                                                                                     
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))     

    textSizeWidth, textSizeHeight = textSize        

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))        
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))         

class Application(tk.Frame):
    def __init__(self, master=None):       
        super().__init__(master)        # gọi hàm init cha
        self.master.title("Đọc biển số")       #tên form
        self.master.minsize(1000,520)   #kích cỡ tối thiểu
        self.master = master
        self.pack()                     #gọi hàm tạo widgets (vật dụng)
        self.create_canvas()            #khởi tạo canvas - khung chứa ảnh
        self.create_widgets()           #khởi tạo phụ kiện (button)
        self.currentImage = " "         #hình cuối


    def create_widgets(self):
        self.btnSpace = Button(master = self, text = " ", padx=80) 
        self.btnSpace.pack(side = "top")                               

        self.btnGetImage = Button(master = self, text = "Chọn ảnh") 
        self.btnGetImage["command"] = self.browseFiles                  
        self.btnGetImage.pack(side = "top")                        

        self.space = Canvas(master = self, width = 140, height = 30)                               
        self.space.pack(side="top")

        self.btnExecute = Button(master = self, text = "Xem biển số")
        self.btnExecute["command"] = self._execute                 
        self.btnExecute.pack(side = "top")                         

        self.result = Text(master = self, height = 1, width = 15, font=("Helvetica", 14))
        self.result.pack(side = "top")

        self.space2 = Canvas(master = self, width = 140, height = 30)                                
        self.space2.pack(side="top")




    def create_canvas(self):
        self.slideImage = LabelFrame(master = self, width = 200, height = 500, text="Danh sách ảnh")
        self.slideImage.pack_propagate(0)
        self.slideImage.pack(side="left")

        self.mainImage = Canvas(master = self, width = 700, height = 500, background = "Gray")      
        self.mainImage.pack(side="left")                                                           

    def browseFiles(self):
        self.getListImage = filedialog.askopenfilenames(initialdir = ".", title = "Select a File", filetypes = (("all files", "*.*"), ("Text files", "*.txt*")))
        listImage = list(self.getListImage)
        self.LoadImageList(self.getListImage)
        self.LoadImage(self.getListImage[0])

    def LoadImageList(self, listImage):
        for widget in self.slideImage.winfo_children():
            widget.destroy()

        vscrollbar = Scrollbar(self.slideImage)
        vscrollbar.pack( side = RIGHT, fill = Y )

        hscrollbar = Scrollbar(self.slideImage, orient = 'horizontal')
        hscrollbar.pack( side = BOTTOM, fill = X )

        myList = Listbox(master = self.slideImage, width = 200, height = 500, yscrollcommand = vscrollbar.set, xscrollcommand = hscrollbar.set)
        myList.pack( side = LEFT, fill = BOTH )
        for i in range (len(listImage)):
            myList.insert(tk.END, listImage[i])

        vscrollbar.config( command = myList.yview )
        hscrollbar.config( command = myList.xview )

        def onselect(event):
            w = event.widget
            idx = int(w.curselection()[0])
            value = w.get(idx)
            self.LoadImage(value)

        myList.bind('<<ListboxSelect>>', onselect)

    def CreateLinkLabel(self, link):
        label = Label(master = self.slideImage, text = os.path.basename(link), fg="blue", cursor="hand2")
        label.pack(side = "top")
        label.bind("<Button-1>", lambda e : self.LoadImage(link))

    def LoadImage(self, path):
        self.currentImage = path
        img = Image.open(path)                                              
        image = img.resize((700, 500), Image.ANTIALIAS)                       
        self.mainImage.image = ImageTk.PhotoImage(image)                                
        self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')       

    def _execute(self):
        print(self.currentImage)
        main(self.currentImage)

    def changeImage(self, sourceImage):
        image = sourceImage.resize((700, 500), Image.ANTIALIAS)                         
        self.mainImage.image = ImageTk.PhotoImage(image)                                
        self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')      

    def insertText(self, text):
        self.result.delete(1.0,END)
        self.result.insert(tk.END, text, "center")

def a():
    app.LoadImage("Image/1.jpg")

root = tk.Tk()                  
app = Application(master=root)      
app.mainloop()                    


















