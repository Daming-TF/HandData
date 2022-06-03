import os
import numpy as np
# 关于hdf5/lmdb/disk存储图片相关库
from PIL import Image
import pickle
import lmdb
import h5py
import time

# 海量数据集读写测试函数
# ---------------------- disk -------------------------- #
def store_many_disk(images, disk_dir):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
        disk_dir     the path to save the output images
    """
    # num_images = len(images)
    # Save all the images one by one
    for i, image in enumerate(images):
        save_path = os.path.join(disk_dir, f"{i}.png")
        Image.fromarray(image).save(save_path)

    # Save all the labels to the csv file
    """"
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])
    """""
# ----------------------LMDB class-------------------------- #
class CIFAR_Image():
    def __init__(self, image):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        # 序列化
        self.image = image.tobytes()
        # self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        # 反序列化
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

# ----------------------LMDB -------------------------- #
def store_many_lmdb(images, lmdb_dir):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
        lmdb_dir     the path to save the lmdbfile
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    lmdb_path = os.path.join(lmdb_dir, f"{num_images}_lmdb")
    env = lmdb.open(lmdb_path, map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

# ----------------------HDF5-------------------------- #
def store_many_hdf5(images, hdf5_dir):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
        hdf5_dir     the path to save the hdf5file
    """
    print("hdf5 start to work")
    start = time.time()
    num_images = len(images)
    shape = (num_images, images[0].shape)
    hdf5_shape = tuple([shape[0]]+list(shape[1]))
    print(hdf5_shape)
    # Create a new HDF5 file
    h5py_path = os.path.join(hdf5_dir, f"{num_images}_many.h5")
    file = h5py.File(h5py_path, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", hdf5_shape, h5py.h5t.STD_U8BE, data=images)

    ''' meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )'''
    file.close()
    end = time.time()
    print(f'hdf5_detector time: %f' % (end - start))