#!/usr/bin/python

'''
Run over all files which are .npz,
and if they have 3 arrays then open and save as RGB in an npz.
'''


import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

DATA = "data"
LABELS = "labels"

def test_img_shape(path):
    img = imageio.imread(path)
    print("==== > test path {}, shape: {}".format(path, img.shape))



def creat_last_and_first_data_set():
    data_root_path = "C:\\Users\\Eldan\\Documents\\Final Project\\AutomatedFetal_CV_project\\roi_ground_truth"
    data_path = os.path.join(data_root_path, DATA)
    labels_path = os.path.join(data_root_path, LABELS)
    for j, path_name in enumerate([data_path,labels_path]):
        target_path_name ="C:\\Users\\Eldan\\Documents\\Final Project\\AutomatedFetal_CV_project\\last_and_first_slices"
        if not j:
            target_path_name = os.path.join(target_path_name, DATA)

        else:
            target_path_name = os.path.join(target_path_name, LABELS)
        print('------------',path_name)
        print("+++++++++", target_path_name)
        file_list = os.listdir(path_name)
        alraedy_fix = []
        for file in file_list:
            prefix = file.find('frame')
            if file[:prefix] in alraedy_fix:
                continue
            else:
                alraedy_fix.append(file[:prefix])
                full_file_list = [f for f in file_list if f.startswith(file[:prefix])]
                prefixed = [f[f.find('frame'):f.rfind('.')] for f in full_file_list if f.startswith(file[:prefix])]
                frame_number = [int(f[f.find('e')+1:]) for f in prefixed if f[f.find('e')+1:].find('_') == -1]
                full_path_min = os.path.join(path_name, full_file_list[np.argmin(frame_number)])
                full_path_max = os.path.join(path_name, full_file_list[np.argmax(frame_number)])

                target_path_first = os.path.join(target_path_name, full_file_list[np.argmin(frame_number)][:prefix]+'frame'+ str(np.min(frame_number)-1))

                target_path_last = os.path.join(target_path_name, full_file_list[np.argmax(frame_number)][:prefix]+'frame'+str(np.max(frame_number)+1))
                for i,file_name in enumerate([full_path_min,full_path_max]):
                    obj = np.load(file_name)
                    if len(obj.keys()) == 3:
                        arr_0, arr_1, arr_2 = obj['arr_0'], obj['arr_1'], obj['arr_2']
                        if arr_0.shape[1] != 512:
                            print('found a problem')
                            continue
                        new_arr = np.zeros((arr_0.shape[0], arr_0.shape[1], 3))
                        if i:
                            new_arr[:, :, 0] = arr_1
                            new_arr[:, :, 1] = arr_2
                            np.savez(target_path_last, new_arr[:,:,0], new_arr[:,:,1], new_arr[:,:,2])
                        else:
                            new_arr[:, :, 1] = arr_0
                            new_arr[:, :, 2] = arr_1
                            np.savez(target_path_first,new_arr[:,:,0], new_arr[:,:,1], new_arr[:,:,2])
                    # a,b,c = new_arr[:, :, 0],new_arr[:, :, 1],new_arr[:, :, 2]
                    # plt.imshow(a,cmap ='gray')
                    # plt.figure()
                    # plt.imshow(b,cmap ='gray')
                    # plt.figure()
                    # plt.imshow(c,cmap ='gray')
                    # plt.show()
            print(np.argmin(prefixed))
            print(np.argmax(prefixed))
            print(target_path_last)
            print(target_path_first)
            # np.savez(target_path_name, new_arr[0], new_arr[1], new_arr[2]))


def handle_file(path, full_path, new_path):
    # print("==> Handle file: {}".format(path))
    obj = np.load(full_path)
    if len(obj.keys()) == 3:
        arr_0, arr_1, arr_2 = obj['arr_0'], obj['arr_1'], obj['arr_2']
        if "gt" in new_path:
            new_arr = arr_1
            # plt.figure()
            # plt.imshow(new_arr,cmap ='gray')
            # plt.show()
        else:
            new_arr = np.zeros((arr_0.shape[0], arr_0.shape[1], 3))
            new_arr[:, :, 0] = arr_0
            new_arr[:, :, 1] = arr_1
            new_arr[:, :, 2] = arr_2
            # a,b,c = new_arr[:, :, 0],new_arr[:, :, 1],new_arr[:, :, 2]
            # plt.imshow(a,cmap ='gray')
            # plt.figure()
            # plt.imshow(b,cmap ='gray')
            # plt.figure()
            # plt.imshow(c,cmap ='gray')
            # plt.show()
        imageio.imsave(new_path, new_arr)
        # print("     Saved to {}".format(full_path))


def preprocess_files(data_path,labels_path, data_new_path_train,labels_new_path_train,data_new_path_test,labels_new_path_test):
    data_list = os.listdir(data_path)

    for i, f in enumerate(data_list):
        print(f)
        if i % 10 == 0:
            full_path_data = os.path.join(data_path, f)
            full_path_label = os.path.join(labels_path, f)
            new_f_data = os.path.join(data_new_path_test, f)
            new_f_label = os.path.join(labels_new_path_test, f)
        else:
            full_path_data = os.path.join(data_path, f)
            full_path_label = os.path.join(labels_path, f)
            new_f_data = os.path.join(data_new_path_train, f)
            new_f_label = os.path.join(labels_new_path_train, f)

        print(new_f_data)
        handle_file(f, full_path_data, new_f_data)
        print(new_f_label)
        print('-------')
        handle_file(f, full_path_label, new_f_label)



def validate_test_gt():
    test_img_path = os.getcwd() + "\\test\\img\\0"
    test_gt_path = os.getcwd() + "\\test\\gt\\0"
    train_gt_path = os.getcwd() + "\\train\\gt\\0"
    for f in os.listdir(test_img_path):
        gt_new_path = os.path.join(test_gt_path, f)
        if not os.path.exists(gt_new_path):
            old_path = os.path.join(train_gt_path, f)
            os.rename(old_path, gt_new_path)


def main_preprocess():
    root_path = os.getcwd()
    data_root_path = "C:\\Users\\Eldan\\Documents\\Final Project\\AutomatedFetal_CV_project\\roi_ground_truth"

    data_path = os.path.join(data_root_path, DATA)
    data_new_path_train = os.path.join(root_path, "train1", "img", "0")
    data_new_path_test = os.path.join(root_path, "test1", "img", "0")

    labels_path = os.path.join(data_root_path, LABELS)
    labels_new_path_train = os.path.join(root_path, "train1", "gt", "0")
    labels_new_path_test = os.path.join(root_path, "test1", "gt", "0")
    preprocess_files(data_path,labels_path, data_new_path_train,labels_new_path_train,data_new_path_test,labels_new_path_test)


if __name__ == '__main__':
    main_preprocess()
    # validate_test_gt()
    # creat_last_and_first_data_set()
    print('empty')
