# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:55:07 2019
@author: Administrator
"""
import requests
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import re
from PIL import Image, ImageTk


def Fonts(*args):
    font = {'连笔手写': '396', '仿宋': '331', '艺术体': '364', '明星手写': '5','一笔签':'901'}
    return font[comboxlist1.get()]


def Get_sign():
    startUrl = 'http://www.yishuzi.com/b/re13.php'
    name = entry.get()
    name = name.strip()
    if name == "":
        messagebox.showinfo("提示", "请输入您的大名")
    else:
        fontsIndex = Fonts()
        # 模拟浏览器的post数据
        data = {
            'id': name,
            'idi': 'jiqie',
            'id1': fontsIndex,
            'id2': '#FFFDFA',
            'id3': '',
            'id4': '#FF0010',
            'id5': '',
            'id6': '$FF7519'
        }
        result = requests.post(startUrl, data=data)
        result.encoding = 'utf8'
        html = result.text
        print(html)
        pat = '<img src="(.*?)">'
        pat = re.compile(pat)
        print(pat)
        imgPath = pat.findall(html)

        response = requests.get(imgPath[0]).content
        with open('{}.gif'.format(name), 'wb') as f:
            f.write(response)

        bm6 = ImageTk.PhotoImage(file='{}.gif'.format(name))
        Label2 = Label(root, image=bm6)
        Label2.bm = bm6
        Label2.grid(row=4, columnspan=5)  # columnspan指的是组件所跨越的列数


if __name__ == '__main__':
    root = Tk()
    root.title("大师设计")
    root.geometry("600x500")
    #    root.geometry('+400+200')
    label = Label(root, text='姓名', font=('微软雅黑', 20), fg='black')  # 用来显示文本和位图
    label.grid()
    entry = Entry(root, font=('宋体', 25))  # entry输入控件，entry属于root
    entry.grid(row=1, column=0)

    label = Label(root, text='字体', font=('微软雅黑', 10), fg='red')
    label.grid(row=2, column=0, sticky=W)
    comvalue = StringVar()
    comboxlist1 = ttk.Combobox(root, textvariable=comvalue)
    comboxlist1["values"] = ('连笔手写', '仿宋', '明星手写', '艺术体','一笔签')
    comboxlist1.current(0)  # 选择第一个，这里的0表示comboxlist1["values"]的下标0，
    comboxlist1.bind("<<ComboboxSelected>>", Fonts)  # 绑定thinter下拉框事件（Combobox）绑定Fonts函数
    comboxlist1.grid(row=2, column=1, sticky=W)

    button = Button(root, text="设计签名", font=('w微软雅黑', 20), command=Get_sign)
    button.grid(row=3, column=0, sticky=W)
    root.mainloop()  # 死循环，让窗口一直显示

'''
    root.pack()#显示
  默认的控件在窗口中的对齐方式是居中。可以使用sticky选项去指定对齐方式，可以选择的值有：N/S/E/W，
  分别代表上对齐/下对齐/左对齐/右对齐，可以单独使用N/S/E/W，也可以上下和左右组合使用，达到不同的对齐效果

  tkinter资料
  https://blog.csdn.net/sofeien/article/details/50982208
  https://blog.csdn.net/sofeien/article/details/49444001
  http://effbot.org/tkinterbook/frame.htm

'''
