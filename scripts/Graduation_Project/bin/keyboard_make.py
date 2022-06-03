keys = [['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L',';'],
        ['Z','X','C','V','B','N','M',',','.','/']]
#Button的主要目的就是为了将位置，大小，内容link在一起，方便后面使用
class Button():
    def __init__(self,pos,text,size = [50,50]):
        self.pos = pos
        self.text = text
        self.size = size
#创建由Button对象组成的List
for j in range(len(keys)):
    for x,key in enumerate(keys[j]):
        buttonList.append(Button([60*x+20,100+j*60],key))
#在图像流上绘制键盘
def drawAll(img,buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        #绘制矩形底框，填充色为RGB（255, 0, 255）
        cv2.rectangle(img, but，ton.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        #按键盘布局排列keys
        cv2.putText(img, button.text, (x + 10, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img
