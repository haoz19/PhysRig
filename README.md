# GitHub工作流程

本文旨在建立一个标准的工作流程，便于查看历史版本、分布式工作、以及管理工作流。


## 基本流程

面对一个新的project，首先在GitHub上建立一个private仓库，仓库名称设置为project名称，上传该markdown文档至main分支，**保留main分支不变！**。

我们通过GitHub的分支来管理较大的版本变动，分支名设置为该版本的特点；在每一个版本（分支）下通过不同的文件夹管理较小的版本变动。

每次开始工作之前，切换到对应的分支，并pull最新版本。改动完成后push至该分支。

以下是常用的GitHub命令：

工作前获取变化：
```
GIT_SSH_COMMAND='ssh -i /u/haozhang/files/physdreamer/ssh/id_rsa' # 关于涉及远程仓库的指令需要带自己的私钥

passphrase: xhl0923

git clone git@github.com:jamesdemon923/PhysRig.git

git remote add origin <远程仓库地址>  # 链接远程仓库，创建主分支

git pull origin main  # 获取远程仓库的变化
```

工作后上传变化：
```
git init  # 初始化仓库，每次提交变化之前都需要做

git add .(文件name)  # 添加文件到待提交区域

git commit -m "first commit"  # 添加文件描述信息

git push -u origin main  # 把本地仓库的文件推送到远程仓库main分支

# 我们可以用git push来创建远程分支。我们创建一个新分支，然后直接git push到相同的新名字。
```

其他命令：
```
git branch -a  # 查看所有分支以及当前所在分支

git branch -m new-name  # 将当前分支的名字改为new-name

git branch [branch-name]  # 新建一个分支，并切换到该分支

git checkout [branch-name]  # 切换到目标分支

git branch -d [branch-name]  # 删除本地分支

git push origin --delete [branch-name]  # 删除远程分支
```


## 文件目录格式

在main分支中，我们将只有一个文件，即README.md

在其他分支中，格式如下：
```
- sub-version1
    - ThridParty_Codebase
    - Codebase
        - model
        - dataloader.py
        ...
    - Output
        - Experiment_name1
            - logs
        ...
- sub-version2
- sub-version3
- sub-version4
...
```


## 本地文件管理

由于GitHub并不能上传太大的文件，这就导致我们的checkpoint以及某些中间保留的数据只能保存在本地。所以，我们也需要一个本地文件管理格式：
```
- GitHub仓库名（连接到远程GitHub）
    - （根据上面说的branch+文件夹控制）
- Output（本地保存log与数据，不远程连接到GitHub）
    - branch名（版本名）
        - Experiment_name1
            - logs
            - data
            ...
```
我们在实验输出log的时候就记得同时在两个地方输出。