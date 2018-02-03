# encoding: utf-8

import cv2
import numpy as np
import random
import os
import math
import time
import screencap

# def pull_screenshot():
#     os.system('adb shell screencap -p /sdcard/autojump.png')
#     os.system('adb pull /sdcard/autojump.png .')

def find_location():
    img_playground = cv2.imread('autojump.png', 1)
    img_player     = cv2.imread('template_player.png', 1)
    height, width = img_playground.shape[:2]
    h, w = img_player.shape[:2]

    # 模板匹配截图中小跳棋的位置
    res = cv2.matchTemplate(img_playground, img_player, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    left_top = max_loc
    right_bottom = (left_top[0]+w, left_top[1]+h)
    cv2.rectangle(img_playground, left_top, right_bottom, (255,255,255), 2)

    center1_loc = (left_top[0] + 45, left_top[1] + 215)
    


    # 使用边缘检测匹配物块上沿
    img_rgb = cv2.GaussianBlur(img_playground, (5, 5), 0)
    canny_img = cv2.Canny(img_rgb, 1, 10)
    H, W = canny_img.shape

    # 消去小跳棋轮廓对边缘检测结果的干扰
    for k in range(left_top[1] - 50, left_top[1] + 210):
        for b in range(left_top[0] - 130, left_top[0] + 115):
            canny_img[k][b] = 0 # 黑色

    img_rgb, x_center, y_center = get_center(canny_img, H)


    cv2.circle(img_rgb, center1_loc, 10, (255,255,255), -1)
    cv2.circle(img_rgb, (x_center, y_center), 10, (255,255,255), -1)

    cv2.imwrite('last.png', img_rgb)

    return img_rgb, center1_loc[0], center1_loc[1], x_center, y_center


def get_center(img_canny, H):
    # 利用边缘检测的结果寻找物块的上沿和下沿
    # 进而计算物块的中心点
    y_top = np.nonzero([max(row) for row in img_canny[500:]])[0][0] + 500
    x_top = int(np.mean(np.nonzero(img_canny[y_top])))

    y_bottom = y_top + 150 # 应对花纹木纹等非纯色平面，默认一个至少150的高
    for row in range(y_bottom, H):
        if img_canny[row, x_top] != 0:
            y_bottom = row
            break

    x_center, y_center = x_top, (y_top + y_bottom) // 2
    return img_canny, x_center, y_center


def jump(distance):
    print('distance: %s' % distance)
    press_time = int(distance * 1.35)

    rand = random.randint(0, 9) * 10
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {time}'.format(x1=750+rand, y1=1500+rand, x2=750+rand, y2=1500+rand, time=press_time)
    print(cmd)
    os.system(cmd)



def main():
    screencap.check_screencap()

    while True:
        screencap.pull_screencap()

        img, piece_x, piece_y, board_x, board_y = find_location()

        distance = math.sqrt((piece_x-board_x) ** 2 + (piece_y-board_y) ** 2)

        jump(distance)

        time.sleep(random.uniform(0.9, 1.2))

        h, w = img.shape[:2]
        res = cv2.resize(img, (int(w/3), int(h/3)), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('frame', res)
        if cv2.waitKey(1) == ord('q'):
            break

# img_playground = cv2.resize(img_playground, (int(width/3), int(height/3)), interpolation = cv2.INTER_CUBIC)

# cv2.imshow('wechat jump', img_playground)
# cv2.waitKey(0)

if __name__== '__main__':
    main()
