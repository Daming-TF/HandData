这里主要是一些与json文件相关的工具文件：
1.  check_json(21pts).py：
	从我们定义好的coco格式json文件中调取关键点信息，并打印到对应图片上

2. combine_json.py:
	把不同开源数据集转化后的json文件结合成一个

3. json_struction.py：
	可以查看不同格式json文件的设定格式

3. sort_ImgAnno_json.py:
	有些json文件中，Img和anno中的id对应不上，通过该程序可以重新排序

4. statistics_json.py:
	统计json中每个范围的图片数量，看看和理论上的数值是否对应上

5. update_json.py:
	剔除json中badcase信息