# Tools——工具库板块
## 目录
- [Json Tools——json工具库](#Json_Tools——json工具库)
- [Mark Tools——标记工具库](#Mark_Tools——标记工具库) 
- [Visualization Tools——可视化工具库](#Visualization-Tools——可视化工具库) 
- [Vedio Tools——视频工具库](#Others——其他工具) 


## Json_Tools——json工具库
        json工具模块主要是用来处理原始数据或者训练数据的json文件；
- **combine_json.py**  
```Note```: 用于把多个json文件的内容合并 


- **combine_txt.py**  
```Note```: 用于把多个txt文件的内容合并


- **cut_from_data.py**  
```Note```: 指定json文件路径，把该json数据中满足自定义的image id范围数据抽离出来

  
- **Inspection_data.py**  
```Note```: 检查指定的训练json文件所有训练数据路径是否存在


- **json_struction.py**  
```Note```: 可视化json文件结构


- **sort_ImgAnno_json.py**  
```Note```: json数据排序


- **statistics_json.py**  
```Note```: 统计json数据的分布（每个数据集的数量）


- **update_json(whole-body).py**  
```Note```: 根据badcase.txt更新原始全图的json训练数据


- **update_json.py**  
```Note```: 根据badcase.txt更新剪裁图片的json训练数据


## Mark_Tools——标记工具库
- **left_label_data**  
```Note```: left_hand_label_marktools.py的威力加强版，支持重标数据和无效数据标签的记录，并把结果保存到txt文档


- **face_vis_marker.py**  
```Note```: 手动标记左右眼，鼻子，嘴巴四个关键点的可见性(0/1)


- **left_hand_label_marktools.py**  
```Note```: 手动修改json数据中心左右手标签，支持记录时间和当前任务状态，下一次记录标记时读取存档


- **manual_clean_tool.py**  
```Note```: left_hand_label_marktools.py的威力加强版，支持重标数据和无效数据标签的记录，并把结果保存到txt文档


## Visualization-Tools——可视化工具库
### 模型参数和运算量对比可视化
通过读取excel把每个训练模型的params，MFlops，F1，Model，Resolution，Alpha，Decoder信息读取并画到图标上 ，效果如下图：  
![image](https://github.com/Daming-TF/HandData/blob/master/material/plot.jpg)  


### 训练数据/标注数据可视化
- **check_all_data(whole_body).py**  
```Note```: 通用的训练数据可视化接口,效果如下图  
- ![image](https://github.com/Daming-TF/HandData/blob/master/material/%E5%8F%AF%E8%A7%86%E5%8C%96.jpg) 

- **check_batch_data.py**  
```Note```: 对比标注团队反馈的batch data覆盖前后数据差别，用于检查标注团队标注质量



## Others——其他工具
- **vedio_writer.py**  
```Note```: 用于录屏的工具


- **video_frame_extraction.py**  
```Note```: 视频抽帧工具