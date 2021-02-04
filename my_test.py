import cv2
import numpy as np


def OnMouseAction(event, x, y, flags, param):
    global img

    if not hasattr(OnMouseAction, 'position1'):
        OnMouseAction.position1 = 0
    if not hasattr(OnMouseAction, 'position2'):
        OnMouseAction.position2 = 0

    image = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键
        OnMouseAction.position1 = (x, y)  # 获取鼠标的坐标(起始位置)

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:  # 按住左键拖曳不放开
        cv2.rectangle(image, OnMouseAction.position1, (x, y), (0, 255, 0), 3)  # 画出矩形选定框
        cv2.imshow('image', image)

    elif event == cv2.EVENT_LBUTTONUP:  # 放开左键
        OnMouseAction.position2 = (x, y)  # 获取鼠标的最终位置
        cv2.rectangle(image, OnMouseAction.position1, OnMouseAction.position2, (0, 0, 255), 3)  # 画出最终的矩形
        cv2.imshow('image', image)

        min_x = min(OnMouseAction.position1[0], OnMouseAction.position2[0])  # 获得最小的坐标，因为可以由下往上拖动选定框
        min_y = min(OnMouseAction.position1[1], OnMouseAction.position2[1])
        width = abs(OnMouseAction.position1[0] - OnMouseAction.position2[0])  # 切割坐标
        height = abs(OnMouseAction.position1[1] - OnMouseAction.position2[1])

        cut_img = img[min_y:min_y + height, min_x:min_x + width]
        cv2.imshow('Cut', cut_img)


# def test1(dic1):
#     dic1["a"] = 2
#     dic1["b"] = 3


def main():
    global img
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', OnMouseAction)
    cv2.imshow('image', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    # dic = {
    #     "a": 1,
    #     "b": 2
    # }
    # test1(dic)
    # print(dic["a"])
    # print(dic["b"])


if __name__ == '__main__':
    main()