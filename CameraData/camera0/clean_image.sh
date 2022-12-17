#！ /bin/bash
for file in `ls ./`
do
    echo "./"$file
    if [ -d "./"$file"/" ];then
        rm -r "./"$file"/"*".jpg";
        if [ $? -eq 0 ];then
            echo "删除文件夹$file下图片";
        fi
    else
        echo "不是文件夹"
    fi
done
