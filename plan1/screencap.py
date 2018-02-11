# encoding: utf-8

import os
import subprocess
import sys
from PIL import Image


# SCREENSHOT_WAY 是截图方法，经过 check_screenshot 后，会自动递减，不需手动修改
SCREENSHOT_WAY = 3


def pull_screencap():
    """
    获取屏幕截图，目前有 0 1 2 3 四种方法，未来添加新的平台检测方法时，
    可根据效率及适用性由高到低排序
    """
    global SCREENSHOT_WAY
    if 1 <= SCREENSHOT_WAY <= 3:
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        binary_screenshot = process.stdout.read()
        if SCREENSHOT_WAY == 2:
            binary_screenshot = binary_screenshot.replace(b'\r\n', b'\n')
        elif SCREENSHOT_WAY == 1:
            binary_screenshot = binary_screenshot.replace(b'\r\r\n', b'\n')
        fp = open('autojump.png', 'wb')
        fp.write(binary_screenshot)
        fp.close()
    elif SCREENSHOT_WAY == 0:
        os.system('adb shell screencap -p /sdcard/autojump.png')
        os.system('adb pull /sdcard/autojump.png .')


def check_screencap():
    """
    检测获取截图的方式
    """
    global SCREENSHOT_WAY
    if os.path.isfile('autojump.png'):
        try:
            os.remove('autojump.png')
        except Exception:
            pass
    if SCREENSHOT_WAY < 0:
        print('暂不支持当前设备')
        sys.exit()

    pull_screencap()
    try:
        Image.open('autojump.png').load()
        print('采用方式 {} 获取截图'.format(SCREENSHOT_WAY))
    except Exception:
        SCREENSHOT_WAY -= 1
        check_screencap()

