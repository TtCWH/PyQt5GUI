import  numpy as np
import random
import glob
import os.path
import Parameters
import shutil
from PIL import Image

def picture_process(old_path):
    try:
        if not os.path.exists(Parameters.TRAININGDATABASE):
            os.mkdir(Parameters.TRAININGDATABASE)

        new_path = os.path.join(Parameters.TRAININGDATABASE, os.path.basename(old_path))
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        sub_dirs = [x[0] for x in os.walk(old_path)]
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            pic_save_path = os.path.join(new_path, os.path.basename(sub_dir))
            if not os.path.exists(pic_save_path):
                os.mkdir(pic_save_path)

            pic_lists = [x[2] for x in os.walk(sub_dir)]
            for pic_list in pic_lists:
                for pic in pic_list:
                    pic_exists_path = os.path.join(sub_dir, pic)
                    if pic.split('.')[1] == 'tif' or pic.split('.')[1] == 'tiff' or pic.split('.')[1] == 'png'\
                        or pic.split('.')[1] == 'TIF' or pic.split('.')[1] == 'TIFF' or pic.split('.')[1] == 'PNG':
                        im = Image.open(pic_exists_path)
                        new_image_name = os.path.join(new_path, sub_dir, pic.split('.')[0]+'.jpg')
                        im.save(new_image_name)
                    elif pic.split('.')[1] == 'jpg' or pic.split('.')[1] == 'JPG' or pic.split('.')[1] == 'jpeg'\
                        or pic.split('.')[1] == 'JPEG':
                        shutil.copy(pic_exists_path, pic_save_path)
        return new_path
    except:
        return -1


def create_image_lists(testing_percentage, validation_percentage):
    try:
        new_path = picture_process(Parameters.INPUT_DATA)
        bottleneck_path = os.path.join(Parameters.CACHE_DIR, os.path.basename(new_path))
        if not os.path.exists(bottleneck_path):
            os.makedirs(bottleneck_path)

        if bottleneck_path != Parameters.BOTTLENECK_PATH:
            Parameters.BOTTLENECK_PATH = bottleneck_path
    except:
        return -1

    if new_path == -1:
        return -1

    try:
        result = {}
        label_name_list = []
        sub_dirs = [x[0] for x in os.walk(new_path)]
        #print(sub_dirs)
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            for extension in extensions:
                file_glob = os.path.join(new_path, dir_name, '*.' + extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list: continue

            label_name = dir_name.lower()

            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)

                chance = np.random.randint(100)
                if chance < validation_percentage:
                    validation_images.append(base_name)
                elif chance < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)

            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result, new_path
    except:
        return -1


def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = image_dir + '/' + sub_dir + '/' + base_name
    #os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, Parameters.BOTTLENECK_PATH, label_name, index, category) + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor:image_data})

    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_or_create_bottleneck(sess,  image_lists, Set_Path, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(Parameters.BOTTLENECK_PATH, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    #print(bottleneck_path)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, Set_Path, label_name, index, category)
        #print(image_path)
        try:
            image_data = open(image_path, 'rb').read()
            #image_data = gfile.FastGFile(image_path, 'rb').read()
            #print(image_data)
            # image = tf.read_file(image_path)
            # image_data = tf.image.decode_jpeg(image, channels=3)

            bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
        except:
            pass
    else:
        #print("WANGWANGWANG")
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, Set_Path, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name_list = list(image_lists.keys())
        label_name_list.sort()
        label_name = label_name_list[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess,  image_lists, Set_Path, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        #print(bottleneck)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, Set_Path, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    label_name_list.sort()
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, Set_Path, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths



