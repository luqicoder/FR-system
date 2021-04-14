import os
import struct
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import Menu

import cv2
from PIL import Image, ImageTk
from face_recognition import face_recognition


class MainWindows(tk.Tk):
    def __init__(self):
        super().__init__()  # 初始化基类

        self.title("人脸识别系统")
        self.resizable(width=False, height=False)
        self.minsize(640, 320)

        self.tabControl = ttk.Notebook(self)  # Create Tab Control
        self.tab1 = ttk.Frame(self.tabControl)  # Create a tab
        self.tab2 = ttk.Frame(self.tabControl)  # Add a second tab
        self.tab3 = ttk.Frame(self.tabControl)  # Add a second tab

        self.menu_bar = Menu(self)  # Creating a Menu Bar

        self.init_ui()

        self.selected_files = []  # 被选中的文件，获取识别结果被使用
        self.photo_libs = []  # 本地图片库
        self.feature_libs = []  # 本地特征向量库
        self.lib_path = './images/libs'  # 本地库文件路径

        self.face_recog = face_recognition()

        self.update_treeview()

    def init_ui(self):
        # self.btn = tk.Button(self, text='点我吧')
        # self.btn.pack(padx=200, pady=30)
        # self.btn.config(command=self.tell_you)
        self.tabControl.add(self.tab1, text='人脸验证(1:1)')  # Add the tab
        self.tabControl.add(self.tab2, text='人脸辨别(1:N)')  # Make second tab visible
        self.tabControl.add(self.tab3, text='人脸数据库管理')  # Make second tab visible
        self.tabControl.pack(expand=1, fill="both")  # Pack to make visible

        self.init_tab1()
        self.init_tab2()
        self.init_tab3()

        self.config(menu=self.menu_bar)
        self.init_menu()

    def init_tab1(self):
        mighty = ttk.LabelFrame(self.tab1, text='')
        mighty.pack()
        self.label1 = tk.Label(mighty, text='检测图片1', bg="Silver", padx=15, pady=15)
        self.label2 = tk.Label(mighty, text='检测图片2', bg="Silver", padx=15, pady=15)
        self.label1.grid(column=0, row=0, sticky='W')
        self.label2.grid(column=1, row=0, sticky='W')
        btn1 = ttk.Button(mighty, text="选择文件", command=self.select_btn_tab1)
        btn2 = ttk.Button(mighty, text="获取结果", command=self.get_result1)
        btn1.grid(column=0, row=1, sticky='W')
        btn2.grid(column=0, row=2, sticky='W')
        label3 = tk.Label(mighty, text='相似度(一般70%以上可以表示为同一个人):')
        label3.grid(column=0, row=3, sticky='W')
        self.name = tk.StringVar()
        name_entered = ttk.Entry(mighty, width=12, textvariable=self.name)
        name_entered.grid(column=1, row=3, sticky='W')  # align left/West

    def init_tab2(self):
        mighty2 = ttk.LabelFrame(self.tab2, text='')
        mighty2.pack()

        self.label_rec = tk.Label(mighty2, text='待识别图片', bg="Silver", padx=15, pady=15)
        self.label_rec.grid(column=0, row=0, sticky='W')
        btn_sel = ttk.Button(mighty2, text="选择文件", command=self.select_btn_tab2)
        btn2_res = ttk.Button(mighty2, text="获取结果", command=self.get_result2)
        btn_sel.grid(column=0, row=1, sticky='W')
        btn2_res.grid(column=0, row=2, sticky='W')
        label_res = tk.Label(mighty2, text='识别对象名：')
        label_res.grid(column=0, row=3, sticky='W')
        self.name2 = tk.StringVar()
        name_entered2 = ttk.Entry(mighty2, width=12, textvariable=self.name2)
        name_entered2.grid(column=1, row=3, sticky='W')  # align left/West

    def init_tab3(self):
        mighty3 = ttk.LabelFrame(self.tab3, text='本地数据')
        mighty3.pack()
        self.tree = ttk.Treeview(mighty3, height=4, columns=('姓名'))  # 表格
        self.tree.grid(row=0, column=0, sticky='nsew')
        # Setup column heading
        self.tree.heading('0', text='姓名', anchor='center')
        self.tree.column('姓名', anchor='center', width=100)
        s = ttk.Style()
        s.configure('Treeview', rowheight=110)
        s.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Calibri', 11))  # Modify the font of the body
        s.configure("mystyle.Treeview.Heading", font=('Calibri', 13, 'bold'))  # Modify the font of the headings
        s.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})])  # Remove the borders
        newb1 = ttk.Button(mighty3, text='点击添加', width=20, command=self.add_data)
        newb1.grid(row=1, column=0, sticky='nsew')
        newb2 = ttk.Button(mighty3, text='选中删除', width=20, command=self.delete_cur_treeview)
        newb2.grid(row=2, column=0, sticky='nsew')
        newb3 = ttk.Button(mighty3, text='刷新界面', width=20, command=self.update_treeview)
        newb3.grid(row=3, column=0, sticky='nsew')

    def init_menu(self):
        # Add menu items
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="New")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # Add another Menu to the Menu Bar and an item
        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About")
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

    def show_img(self, labels, filename, length=1):
        if len(filename) < 1:
            return
        for i in range(length):
            img = Image.open(filename[i])
            half_size = (256, 256)
            img.thumbnail(half_size, Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            labels[i].configure(image=photo)
            labels[i].image = photo

    def select_file(self):
        self.selected_files = []
        ftypes = [('Image Files', '*.tif *.jpg *.png')]
        dlg = filedialog.Open(filetypes=ftypes, multiple=True)
        filename = dlg.show()
        self.selected_files = filename
        return filename

    def select_btn_tab1(self):
        self.show_img([self.label1, self.label2], self.select_file(), 2)

    def get_result1(self):
        if len(self.selected_files) < 2:
            return

        img_path1 = self.selected_files[0]
        img_path2 = self.selected_files[1]
        # 计算两张图片的余弦相似度
        v1 = self.face_recog.get_single_feature_vector(img_path1)
        v2 = self.face_recog.get_single_feature_vector(img_path2)
        res = self.face_recog.cos_sim(v1, v2).tolist()[0][0]
        res = format(res * 100, '.2f')
        # name.configure(text=str(res[0][0]))
        self.name.set(res + '%')

    def select_btn_tab2(self):
        self.show_img([self.label_rec], self.select_file())

    def get_result2(self):
        if len(self.selected_files) >= 1:
            cur_fea = self.face_recog.get_single_feature_vector(self.selected_files[0])
            max_value = 0
            name = ''
            features_len = len(self.feature_libs)
            for i in range(features_len):
                cur_cos = self.face_recog.cos_sim(self.feature_libs[i][1], cur_fea)
                if cur_cos > max_value:
                    max_value = cur_cos
                    name = self.feature_libs[i][0]
            if max_value < 0.7:
                name = '未识别'
            self.name2.set(name.split('\\')[-1])

    def get_lib(self, suffix='png'):
        ret = []
        for root, dirs, files in os.walk(self.lib_path):
            for file in files:
                # 获取文件名, 文件路径
                suf = file.split('.')[-1]
                if suf == suffix:
                    if suffix == 'png':
                        ret.append([file.split('.')[0], os.path.join(root, file)])
                    elif suffix == 'fea':
                        ret.append(os.path.join(root, file))

        return ret

    def add_data(self):
        files = self.select_file()
        for img_path in files:
            fea = self.face_recog.get_single_feature_vector(img_path)
            save_name = img_path.split('/')[-1].split('.')[0]
            save_name_fea = '.\\images\\libs\\' + save_name + '.fea'
            save_name_img = '.\\images\\libs\\' + save_name + '.png'

            features_points, _ = self.face_recog.get_landmarkAndrect(img_path)
            crop_img = self.face_recog.get_cropImage(features_points[0], img_path)
            cv2.imwrite(save_name_img, crop_img)
            # cv.imshow("1", crop_img)
            # cv.waitKey(0)
            # 参考：https://blog.csdn.net/reyyy/article/details/108223977
            v_size = len(fea)  # 获取列表的长度
            fmt = str(v_size) + 'd'
            with open(save_name_fea, 'wb') as binfile:
                data = struct.pack(fmt, *fea)
                binfile.write(data)

        self.update_treeview()

    # 删除treeview视图中的项，本地数据未修改（待添加）
    def delete_cur_treeview(self):
        item = self.tree.focus()
        self.tree.delete(item)

    def delete_all_treeview(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

    def update_treeview(self):
        self.photo_libs = []
        self.delete_all_treeview()

        libs_img = self.get_lib()
        libs_len = len(libs_img)
        count = 0

        for i in range(libs_len):
            cur_pair = libs_img[i]
            # print(cur_pair)
            img = Image.open(cur_pair[1])
            half_size = (100, 100)
            img.thumbnail(half_size, Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            self.photo_libs.append(photo)
            self.tree.insert('', count, text="", image=self.photo_libs[-1], value=(cur_pair[0]))
            count += 1

        # 更新特征向量库
        self.get_features_vec_lib()

    def get_features_vec_lib(self):
        self.feature_libs = []
        files = self.get_lib('fea')
        for file in files:
            # 读取本地数据
            read_result = []
            fmt = str(512) + 'd'
            with open(file, 'rb') as binfile:
                data = binfile.read()
                a = struct.unpack(fmt, data)
                # print(a)
                read_result.append(a)
            self.feature_libs.append([file.split('/')[-1].split('.')[0], read_result])

    def _quit(self):
        self.quit()
        self.destroy()
        exit()


if __name__ == '__main__':
    app = MainWindows()
    app.mainloop()
