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

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

training = DetectChars.trainingData()
if training == False:                               # if KNN training was not successful
    print("\nerror: KNN traning was not successful\n")  # show error message
    exit()  

###################################################################################################
def main(path):

    imgOriginalScene  = cv2.imread(path)

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if  

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    #cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")
        app.insertText('') # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        if DetectPlates.showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        if DetectPlates.showSteps == True:
            cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        pilImg = cv2.cvtColor(imgOriginalScene, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(pilImg)
        app.changeImage(pilImage)
        app.insertText(licPlate.strChars)

        #cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else

    #cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    #cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
# if __name__ == "__main__":
#     main()

class Application(tk.Frame):
    def __init__(self, master=None):       
        super().__init__(master)        # gọi hàm init cha
        self.master.title("Đọc biển số")       #tên form
        self.master.minsize(1000,520)   #kích cỡ tối thiểu
        self.master = master
        self.pack()                     #gọi hàm tạo widgets (vật dụng)
        self.create_canvas()            #khởi tạo canvas - khung chứa ảnh
        self.create_widgets()           #khởi tạo phụ kiện (button)
        #self.create_phase_result()     #khởi tạo khung hiển thị kết quả cho từng bư
        self.currentImage = " "


    def create_widgets(self):
        self.btnSpace = Button(master = self, text = " ", padx=80)  #thông số button
        self.btnSpace.pack(side = "top")                                #đặt button vào form

        self.btnGetImage = Button(master = self, text = "Chọn ảnh") #thông số button
        self.btnGetImage["command"] = self.browseFiles                  #event click cho button
        self.btnGetImage.pack(side = "top")                         #đặt button vào form

        self.space = Canvas(master = self, width = 140, height = 30)                                  #div khoảng trống
        self.space.pack(side="top")

        self.btnExecute = Button(master = self, text = "Xem biển số") #thông số button
        self.btnExecute["command"] = self._execute                 #event click cho button
        self.btnExecute.pack(side = "top")                         #đặt button vào form

        self.result = Text(master = self, height = 1, width = 15, font=("Helvetica", 14))
        self.result.pack(side = "top")

        self.space2 = Canvas(master = self, width = 140, height = 30)                                  #div khoảng trống
        self.space2.pack(side="top")




    def create_canvas(self):
        self.slideImage = LabelFrame(master = self, width = 200, height = 500, text="Danh sách ảnh")
        self.slideImage.pack_propagate(0)
        self.slideImage.pack(side="left")

        self.mainImage = Canvas(master = self, width = 700, height = 500, background = "Gray")     #thông số canvas   
        self.mainImage.pack(side="left")                                                            #đặt canvas vào form

    def browseFiles(self):
        self.getListImage = filedialog.askopenfilenames(initialdir = ".", title = "Select a File", filetypes = (("all files", "*.*"), ("Text files", "*.txt*")))
        listImage = list(self.getListImage)
        self.LoadImageList(self.getListImage)
        self.LoadImage(self.getListImage[0])
        # img = Image.open(self.getListImage[0])                                                #lấy ảnh từ đường dẫn filename
        # image = img.resize((600, 500), Image.ANTIALIAS)                               #thay đổi kích cỡ ảnh theo canvas
        # self.mainImage.image = ImageTk.PhotoImage(image)                              #gán ảnh đã resize cho canvas image
        # self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')     #đặt ảnh vào canvas

        #self.create_phase_result()
        #self._execute(self.filename)
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
        img = Image.open(path)                                              #lấy ảnh từ đường dẫn filename
        image = img.resize((700, 500), Image.ANTIALIAS)                             #thay đổi kích cỡ ảnh theo canvas
        self.mainImage.image = ImageTk.PhotoImage(image)                                #gán ảnh đã resize cho canvas image
        self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')       #đặt ảnh vào canvas

    def _execute(self):
        print(self.currentImage)
        main(self.currentImage)
        #print(Main.imageFinal)
        # image = Main.imageFinal.resize((700, 500), Image.ANTIALIAS)                             #thay đổi kích cỡ ảnh theo canvas
        # self.mainImage.image = ImageTk.PhotoImage(image)                                #gán ảnh đã resize cho canvas image
        # self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')       #đặt ảnh vào canvas

    def changeImage(self, sourceImage):
        image = sourceImage.resize((700, 500), Image.ANTIALIAS)                             #thay đổi kích cỡ ảnh theo canvas
        self.mainImage.image = ImageTk.PhotoImage(image)                                #gán ảnh đã resize cho canvas image
        self.mainImage.create_image(0,0, image = self.mainImage.image, anchor = 'nw')       #đặt ảnh vào canvas

    def insertText(self, text):
        self.result.delete(1.0,END)
        self.result.insert(tk.END, text, "center")

def a():
    app.LoadImage("Image/1.jpg")

root = tk.Tk()                      #root là tên biến tự đặt, tk là thư viện tkinter, Tk() là hàm tạo form
app = Application(master=root)      
app.mainloop()                      #chạy lặp form


















