import os
import random
import cv2
import numpy as np


class LicensePlateRecognitionBatchLoader(object):
    def __init__(self,
                 license_plate_roi_images_root_dir,
                 batch_size,
                 model_input_image_width,
                 model_input_image_height):
        """
        constructor of data loader

        :param license_plate_roi_images_root_dir: root directory of vehicle license ROI images
        :param batch_size: batch size
        :param model_input_image_width: image's width of model input
        :param model_input_image_height: image's height of model input
        """
        self._license_plate_roi_images_root_dir = license_plate_roi_images_root_dir
        self._batch_size = batch_size
        self._model_input_image_width = model_input_image_width
        self._model_input_image_height = model_input_image_height

        self._samples = list()
        for sub_folder in ['train', 'val', 'test']:
            root_folder = os.path.join(self._license_plate_roi_images_root_dir, sub_folder)
            samples = [os.path.join(root_folder, name) for name in os.listdir(root_folder) if name.endswith('.jpg')]
            self._samples.extend(samples)

        self._train_samples, self._val_samples = self.split(samples=self._samples, train_val_ratio=14.0)

        print('Training Samples = {}'.format(len(self._train_samples)))
        print('Validation Samples = {}'.format(len(self._val_samples)))

        self._train_batch_amount = len(self._train_samples) // self._batch_size
        self._val_batch_amount = len(self._val_samples) // self._batch_size
        print('Training Batches = {}'.format(self._train_batch_amount))
        print('Validation Batches = {}'.format(self._val_batch_amount))

        self._train_batch_index, self._val_batch_index = 0, 0

        # DO NOT MODIFY
        self._provinces = [
            "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
            "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
            "青", "宁", "新", "警", "学", "O"]
        self._alphabets = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        self._ads = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', 'O']

    @staticmethod
    def split(samples, train_val_ratio):
        random.shuffle(samples)
        split_index = int(len(samples) * train_val_ratio / (1.0 + train_val_ratio))
        train_samples = samples[0: split_index]
        val_samples = samples[split_index:]
        return train_samples, val_samples

    def generate_ground_truth(self, image_full_path):
        """
        generate ground truth according to given image full path

        :param image_full_path: input image path
        :return: a dictionary which includes all ground truth information
        """
        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        image_data = cv2.resize(image, (self._model_input_image_width, self._model_input_image_height))
        image_data = image_data[:, :, ::-1]     # convert BGR to RGB
        image_data_float = image_data.astype('float32')
        image_data_float /= 255.0

        image_name = image_full_path.split('/')[-1]
        items = image_name.split('#')
        license_num_str = (items[-2].split('[')[1].split(']')[0])
        license_num_gt_list = [int(index) for index in license_num_str.split('_')]
        readable_license_num_gt = items[-1].split('].jpg')[0].split('[')[1]

        assert len(license_num_gt_list) == 7

        return image_data_float, \
               readable_license_num_gt, \
               license_num_gt_list[0], license_num_gt_list[1], license_num_gt_list[2], license_num_gt_list[3], \
               license_num_gt_list[4], license_num_gt_list[5], license_num_gt_list[6]

    def next_batch(self, phase='Train'):
        """
        get next batch data in given phase

        :param phase: a value in ['train', 'val', 'test']
        :return:
        """
        if phase in ['train', 'Train', 'training', 'Training']:
            samples = self._train_samples[
                      self._train_batch_index * self._batch_size:(1 + self._train_batch_index) * self._batch_size]
            self._train_batch_index += 1
        elif phase in ['val', 'Val', 'validation', 'Validation']:
            samples = self._val_samples[
                      self._val_batch_index * self._batch_size:(1 + self._val_batch_index) * self._batch_size]
            self._val_batch_index += 1
        else:
            raise RuntimeError('No support phase {}'.format(phase))

        image_batch_data = list()
        readable_plate_num_gt_batch_data = list()
        plate_pos_1_gt_batch_data = list()
        plate_pos_2_gt_batch_data = list()
        plate_pos_3_gt_batch_data = list()
        plate_pos_4_gt_batch_data = list()
        plate_pos_5_gt_batch_data = list()
        plate_pos_6_gt_batch_data = list()
        plate_pos_7_gt_batch_data = list()

        for full_path in samples:
            image, gt_str, gt_1, gt_2, gt_3, gt_4, gt_5, gt_6, gt_7 = self.generate_ground_truth(image_full_path=full_path)
            image_batch_data.append(image)
            readable_plate_num_gt_batch_data.append(gt_str)
            plate_pos_1_gt_batch_data.append(gt_1)
            plate_pos_2_gt_batch_data.append(gt_2)
            plate_pos_3_gt_batch_data.append(gt_3)
            plate_pos_4_gt_batch_data.append(gt_4)
            plate_pos_5_gt_batch_data.append(gt_5)
            plate_pos_6_gt_batch_data.append(gt_6)
            plate_pos_7_gt_batch_data.append(gt_7)

        image_batch_data = np.array(image_batch_data)
        plate_pos_1_gt_batch_data = np.array(plate_pos_1_gt_batch_data)
        plate_pos_2_gt_batch_data = np.array(plate_pos_2_gt_batch_data)
        plate_pos_3_gt_batch_data = np.array(plate_pos_3_gt_batch_data)
        plate_pos_4_gt_batch_data = np.array(plate_pos_4_gt_batch_data)
        plate_pos_5_gt_batch_data = np.array(plate_pos_5_gt_batch_data)
        plate_pos_6_gt_batch_data = np.array(plate_pos_6_gt_batch_data)
        plate_pos_7_gt_batch_data = np.array(plate_pos_7_gt_batch_data)
        readable_plate_num_gt_batch_data = np.array(readable_plate_num_gt_batch_data)

        return image_batch_data, plate_pos_1_gt_batch_data, \
               plate_pos_2_gt_batch_data, plate_pos_3_gt_batch_data, plate_pos_4_gt_batch_data, \
               plate_pos_5_gt_batch_data, plate_pos_6_gt_batch_data, plate_pos_7_gt_batch_data, \
               readable_plate_num_gt_batch_data

    def reset(self, phase='Train'):
        """
        After each epoch running, we randomly shuffle the samples order for next round

        :param phase: a value in ['train', 'val', 'test']
        :return:
        """
        if phase in ['train', 'Train', 'training', 'Training']:
            random.shuffle(self._train_samples)
            self._train_batch_index = 0
        elif phase in ['val', 'Val', 'validation', 'Validation']:
            random.shuffle(self._val_samples)
            self._val_batch_index = 0
        else:
            raise RuntimeError('No support phase {}'.format(phase))

    def batch_amount(self, phase='Train'):
        """
        return amount of batches in given phase

        :param phase: a value in ['train', 'val']
        :return:
        """
        if phase in ['train', 'Train', 'training', 'Training']:
            return self._train_batch_amount
        elif phase in ['val', 'Val', 'validation', 'Validation']:
            return self._val_batch_amount
        else:
            raise RuntimeError('No support phase {}'.format(phase))

    @property
    def provinces(self):
        return self._provinces

    @property
    def alphabets(self):
        return self._alphabets

    @property
    def ads(self):
        return self._ads


if __name__ == '__main__':
    data_loader = LicensePlateRecognitionBatchLoader(
        license_plate_roi_images_root_dir='./samples/',
        batch_size=32,
        model_input_image_width=116,
        model_input_image_height=40
    )

