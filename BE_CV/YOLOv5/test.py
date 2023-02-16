#cài đặt thư viện requirements ở terminal của yolov5 trước khi chạy chương trình
#pip install -r requirements.txt
import os, sys
os.chdir("yolov5")
os.system('python detect.py --source 0 --weights ../models/best_v4.pt')
#os.system('python detect.py --source C:/Users/leqh2/Documents/BE_CV/download(1).jpg --weights ../models/best_v4.pt')
