#coding=utf-8  
  
import os
import sys
path = sys.argv[1]
count = 1
for file in os.listdir(path):
    os.rename(os.path.join(path,file),os.path.join(path,str(count)+".jpg"))
    count+=1
