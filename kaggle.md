所有文件的顶级父目录是：/kaggle

新建的notebook文件的路径是：/kaggle/working

/kaggle下默认存在的三个文件夹:lib,input,working
```python
print(os.listdir('../'))
#['lib', 'input', 'working']
print(os.getcwd())
# /kaggle/working

# input:存放训练数据的文件夹
# working:工作路径，主要是我们创建的代码文件的工作目录
```

kaggle中使用cp复制：
```python
!cp -rf ../input/xx.py ./
# 将input中的文件移动到working下
```
kaggle中创建文件夹：
```python
!mkdir 文件夹
或
import os
os.mkdir("文件夹")
```


kaggle新建kernel：
1.notebook：类似Jupyter notebook
2.script：类似pycharm

kaggle上传数据集：
只可读，如果要修改需要重新上传新的数据集；

kaggle输出文件夹：
output一般情况不会保存，关掉页面就会清空；

kaggle下载输出output文件夹下的文件：
```python
import os
os.chdir('/kaggle/working')
print(os.getcwd())
print(os.listdir("/kaggle/working"))
from IPython.display import FileLink
FileLink('output文件下已存在的文件名')
```

kaggle关于保存：
https://www.kaggle.com/discussions/general/224266


kaggle关于重启kernel：
https://www.kaggle.com/discussions/getting-started/53147


save version:
点击save version后会多出一个进程，这个进程不会随着长时间不在电脑前就断掉，为了节省GPU时间，可以关闭原来的进程，只保留new version就行。运行完后就可以随时查看永久保存的output了。此时可以关闭浏览器，程序在kaggle后台运行；
注意：
1.选择`save & run all(commit)`，否则结果不保存；
2.save后开始训练，此时，可以关闭页面。若要留在此页面，记得要把GPU关闭，否则也会扣除时长，加上你训练的服务器，等于扣除了2倍

