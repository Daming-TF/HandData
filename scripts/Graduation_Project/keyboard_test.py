'''
1.链接摄像头
2.识别手势
3.绘制键盘
 3.1创建键盘字母List
 3.2通过循环绘制键盘
4.根据坐标，取得返回字母
 4.1 利用lmList[8]食指之间坐标，判断选中的字母
 4.2 利用食指与中指之间的距离，确认输入的字母
 5.扩展，修改键盘背景
 6.利用pynput模拟真实键盘输入
'''
import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Key,Controller
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
#识别手势
detector = HandDetector(detectionCon=0.8)
keyboard = Controller()
#键盘关键字
keys = [['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L',';'],
        ['Z','X','C','V','B','N','M',',','.','/']]
class Button():
    def __init__(self,pos,text,size = [50,50]):
        self.pos = pos
        self.text = text
        self.size = size
    # def draw(self,img):
    #     x,y = self.pos
    #     w,h = self.size
    #     cv2.rectangle(img, self.pos, (x+w,y+h), (255, 0, 255), cv2.FILLED)
    #     cv2.putText(img, self.text, (x+10,y+40),
    #                 cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 255), 2)
    #     return img
buttonList = []
finalText = ''
for j in range(len(keys)):
    for x,key in enumerate(keys[j]):
        #循环创建buttonList对象列表
        buttonList.append(Button([60*x+20,100+j*60],key))
#mybutton = Button([100,100],"Q")
def drawAll(img,buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img,(x,y,w,h),20,rt = 0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img
# def drawAll(img,buttonList):
#     imgNew = np.zeros_like(img,np.uint8)
#     for button in buttonList:
#         x,y = button.pos
#         w, h = button.size
#         # cvzone.cornerRect(imgNew,(button.pos[0],button.pos[1],button.size[0],button.size[1]),20,rt = 0)
#         cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
#         cv2.putText(img, button.text, (x + 10, y + 40),
#                     cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
#         out = img.copy()
#         alpha = 0.1
#         mask = imgNew.astype(bool)
#         print(mask.shape)
#         out[mask] = cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]
#         return out
while True:
    success,img = cap.read()
    #识别手势
    img = detector.findHands(img)
    lmList,bboxInfo = detector.findPosition(img)

    # img = mybutton.draw(img)
    img = drawAll(img,buttonList )
    if lmList:
        for button in buttonList:
            x,y = button.pos
            w,h = button.size
            if x<lmList[8][0]<x+w and y<lmList[8][1]<y+h:
                cv2.rectangle(img, (x-5,y-5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 10, y + 40),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

                l,_,_ = detector.findDistance(8,12,img,draw=False)
                print('中指(12)和食指(8)之间的距离：',l)
                if l < 30:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                    finalText += button.text
                    print('当前选中的是：', button.text)
                    sleep(0.2)
    cv2.rectangle(img, (20,350), (600, 400), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (20, 390),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
