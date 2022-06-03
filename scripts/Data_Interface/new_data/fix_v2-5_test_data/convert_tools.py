import os
import shutil


def convert_coco_format_for_whole(json_file, image_info, annotation_info, save_dir,
                                  save_flag=1, coco_id=None, image_id=None):
    coco_id = coco_id*2 + (~save_flag+2)
    file_name = str(image_id).zfill(12)+'.jpg'
    image_info['file_name'] = file_name
    image_info['id'] = image_id
    annotation_info['id'] = coco_id
    annotation_info['image_id'] = image_id

    if save_flag:
        data_path = image_info["image_dir"]
        save_path = os.path.join(save_dir, file_name)
        shutil.copyfile(data_path, save_path)
        json_file["images"].append(image_info)
    json_file["annotations"].append(annotation_info)
