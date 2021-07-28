import os
import tensorflow as tf
from license_plate_batch_loader import LicensePlateRecognitionBatchLoader


class LicensePlateRecognizer(object):
    def __init__(self, dataset_root_dir, dump_models_root_dir, batch_size, save_model_every_batches):
        self._dataset_root_dir = dataset_root_dir
        self._dump_models_root_dir = dump_models_root_dir
        self._batch_size = batch_size
        self._save_model_every_batches = save_model_every_batches

        if not os.path.exists(self._dump_models_root_dir):
            os.mkdir(self._dump_models_root_dir)

        self._roi_height = 40
        self._roi_width = 116
        self._roi_channels = 3
        self._prov_num = 34
        self._alphabets_num = 25
        self._ads_num = 35

        self._model = self.build_model()
        self._model.summary()

        self._batch_generator = LicensePlateRecognitionBatchLoader(
            license_plate_roi_images_root_dir=self._dataset_root_dir,
            batch_size=self._batch_size,
            model_input_image_width=self._roi_width,
            model_input_image_height=self._roi_height
        )

        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self._optimizer = tf.keras.optimizers.Adam()

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

    def build_model(self):
        roi_input = tf.keras.Input(shape=(self._roi_height, self._roi_width, self._roi_channels))

        base_conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)(roi_input)
        base_conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)(base_conv_1)
        base_conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)(base_conv_2)
        base_feats = tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME')(base_conv_3)

        # License Plate Position 1
        attention_map_for_p1 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p1 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p1])
        conv_1_for_p1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p1)
        conv_2_for_p1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p1)
        flatten_for_p1 = tf.keras.layers.Flatten()(conv_2_for_p1)
        dense_for_p1 = tf.keras.layers.Dense(units=self._prov_num, activation=None)(flatten_for_p1)
        predicted_license_plate_prob_for_p1 = tf.keras.layers.Softmax()(dense_for_p1)

        # License Plate Position 2
        attention_map_for_p2 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p2 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p2])
        conv_1_for_p2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p2)
        conv_2_for_p2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p2)
        flatten_for_p2 = tf.keras.layers.Flatten()(conv_2_for_p2)
        dense_for_p2 = tf.keras.layers.Dense(units=self._alphabets_num, activation=None)(flatten_for_p2)
        predicted_license_plate_prob_for_p2 = tf.keras.layers.Softmax()(dense_for_p2)

        # License Plate Position 3
        attention_map_for_p3 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p3 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p3])
        conv_1_for_p3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p3)
        conv_2_for_p3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p3)
        flatten_for_p3 = tf.keras.layers.Flatten()(conv_2_for_p3)
        dense_for_p3 = tf.keras.layers.Dense(units=self._ads_num, activation=None)(flatten_for_p3)
        predicted_license_plate_prob_for_p3 = tf.keras.layers.Softmax()(dense_for_p3)

        # License Plate Position 4
        attention_map_for_p4 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p4 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p4])
        conv_1_for_p4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p4)
        conv_2_for_p4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p4)
        flatten_for_p4 = tf.keras.layers.Flatten()(conv_2_for_p4)
        dense_for_p4 = tf.keras.layers.Dense(units=self._ads_num, activation=None)(flatten_for_p4)
        predicted_license_plate_prob_for_p4 = tf.keras.layers.Softmax()(dense_for_p4)

        # License Plate Position 5
        attention_map_for_p5 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p5 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p5])
        conv_1_for_p5 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p5)
        conv_2_for_p5 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p5)
        flatten_for_p5 = tf.keras.layers.Flatten()(conv_2_for_p5)
        dense_for_p5 = tf.keras.layers.Dense(units=self._ads_num, activation=None)(flatten_for_p5)
        predicted_license_plate_prob_for_p5 = tf.keras.layers.Softmax()(dense_for_p5)

        # License Plate Position 6
        attention_map_for_p6 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p6 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p6])
        conv_1_for_p6 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p6)
        conv_2_for_p6 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p6)
        flatten_for_p6 = tf.keras.layers.Flatten()(conv_2_for_p6)
        dense_for_p6 = tf.keras.layers.Dense(units=self._ads_num, activation=None)(flatten_for_p6)
        predicted_license_plate_prob_for_p6 = tf.keras.layers.Softmax()(dense_for_p6)

        # License Plate Position 7
        attention_map_for_p7 = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='SAME', activation=tf.nn.sigmoid)(base_feats)
        attention_for_p7 = tf.keras.layers.Multiply()([base_feats, attention_map_for_p7])
        conv_1_for_p7 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(attention_for_p7)
        conv_2_for_p7 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1_for_p7)
        flatten_for_p7 = tf.keras.layers.Flatten()(conv_2_for_p7)
        dense_for_p7 = tf.keras.layers.Dense(units=self._ads_num, activation=None)(flatten_for_p7)
        predicted_license_plate_prob_for_p7 = tf.keras.layers.Softmax()(dense_for_p7)

        model = tf.keras.Model(
            inputs=roi_input,
            outputs=[
                predicted_license_plate_prob_for_p1,
                predicted_license_plate_prob_for_p2,
                predicted_license_plate_prob_for_p3,
                predicted_license_plate_prob_for_p4,
                predicted_license_plate_prob_for_p5,
                predicted_license_plate_prob_for_p6,
                predicted_license_plate_prob_for_p7
            ],
            name='LicensePlateRecognitionModel')
        return model

    def evaluate(self):
        self._batch_generator.reset(phase='Validation')
        accuracy_over_batches = list()
        loss_over_batches = list()

        for batch_idx in range(self._batch_generator.batch_amount(phase='Validation')):
            batch_image_data, batch_gt_p1, batch_gt_p2, batch_gt_p3, batch_gt_p4, batch_gt_p5, \
            batch_gt_p6, batch_gt_p7, batch_gt_str = self._batch_generator.next_batch(phase='Validation')

            batch_image_data = tf.constant(batch_image_data)
            batch_gt_p1 = tf.constant(batch_gt_p1)
            batch_gt_p2 = tf.constant(batch_gt_p2)
            batch_gt_p3 = tf.constant(batch_gt_p3)
            batch_gt_p4 = tf.constant(batch_gt_p4)
            batch_gt_p5 = tf.constant(batch_gt_p5)
            batch_gt_p6 = tf.constant(batch_gt_p6)
            batch_gt_p7 = tf.constant(batch_gt_p7)

            [pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, pred_p6, pred_p7] = self._model(batch_image_data, training=False)

            loss_p1 = self._loss_fn(y_true=batch_gt_p1, y_pred=pred_p1)
            loss_p2 = self._loss_fn(y_true=batch_gt_p2, y_pred=pred_p2)
            loss_p3 = self._loss_fn(y_true=batch_gt_p3, y_pred=pred_p3)
            loss_p4 = self._loss_fn(y_true=batch_gt_p4, y_pred=pred_p4)
            loss_p5 = self._loss_fn(y_true=batch_gt_p5, y_pred=pred_p5)
            loss_p6 = self._loss_fn(y_true=batch_gt_p6, y_pred=pred_p6)
            loss_p7 = self._loss_fn(y_true=batch_gt_p7, y_pred=pred_p7)
            batch_loss = loss_p1 + loss_p2 + loss_p3 + loss_p4 + loss_p5 + loss_p6 + loss_p7

            pred_index_1 = tf.argmax(pred_p1, axis=-1)  # shape = (batch_size, )
            pred_index_2 = tf.argmax(pred_p2, axis=-1)  # shape = (batch_size, )
            pred_index_3 = tf.argmax(pred_p3, axis=-1)  # shape = (batch_size, )
            pred_index_4 = tf.argmax(pred_p4, axis=-1)  # shape = (batch_size, )
            pred_index_5 = tf.argmax(pred_p5, axis=-1)  # shape = (batch_size, )
            pred_index_6 = tf.argmax(pred_p6, axis=-1)  # shape = (batch_size, )
            pred_index_7 = tf.argmax(pred_p7, axis=-1)  # shape = (batch_size, )

            flag_index_1 = tf.math.equal(x=pred_index_1, y=batch_gt_p1)  # shape = (batch_size, )
            flag_index_2 = tf.math.equal(x=pred_index_2, y=batch_gt_p2)  # shape = (batch_size, )
            flag_index_3 = tf.math.equal(x=pred_index_3, y=batch_gt_p3)  # shape = (batch_size, )
            flag_index_4 = tf.math.equal(x=pred_index_4, y=batch_gt_p4)  # shape = (batch_size, )
            flag_index_5 = tf.math.equal(x=pred_index_5, y=batch_gt_p5)  # shape = (batch_size, )
            flag_index_6 = tf.math.equal(x=pred_index_6, y=batch_gt_p6)  # shape = (batch_size, )
            flag_index_7 = tf.math.equal(x=pred_index_7, y=batch_gt_p7)  # shape = (batch_size, )

            temp_a = tf.math.logical_and(x=flag_index_1, y=tf.math.logical_and(flag_index_2, flag_index_3))
            temp_b = tf.math.logical_and(x=flag_index_4, y=tf.math.logical_and(flag_index_5, flag_index_6))
            logic_and_res = tf.math.logical_and(x=temp_a, y=tf.math.logical_and(temp_b, flag_index_7))
            correct_records_amount = tf.reduce_sum(tf.cast(logic_and_res, tf.float32))
            batch_accuracy = correct_records_amount / self._batch_size

            accuracy_over_batches.append(batch_accuracy)
            loss_over_batches.append(batch_loss)

        return sum(accuracy_over_batches) / len(accuracy_over_batches), sum(loss_over_batches) / len(accuracy_over_batches)

    def train(self, max_epochs):
        """
        train the model

        :param max_epochs: maximum epochs
        :return:
        """
        global_batch_counter = 0

        for epoch in range(max_epochs):
            self._batch_generator.reset(phase='Train')
            for batch_idx in range(self._batch_generator.batch_amount(phase='Train')):
                batch_image_data, batch_gt_p1, batch_gt_p2, batch_gt_p3, batch_gt_p4, batch_gt_p5, \
                batch_gt_p6, batch_gt_p7, batch_gt_str = self._batch_generator.next_batch(phase='Train')

                batch_image_data = tf.constant(batch_image_data)
                batch_gt_p1 = tf.constant(batch_gt_p1)
                batch_gt_p2 = tf.constant(batch_gt_p2)
                batch_gt_p3 = tf.constant(batch_gt_p3)
                batch_gt_p4 = tf.constant(batch_gt_p4)
                batch_gt_p5 = tf.constant(batch_gt_p5)
                batch_gt_p6 = tf.constant(batch_gt_p6)
                batch_gt_p7 = tf.constant(batch_gt_p7)

                with tf.GradientTape() as tape:
                    [pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, pred_p6, pred_p7] = self._model(batch_image_data, training=True)

                    loss_p1 = self._loss_fn(y_true=batch_gt_p1, y_pred=pred_p1)
                    loss_p2 = self._loss_fn(y_true=batch_gt_p2, y_pred=pred_p2)
                    loss_p3 = self._loss_fn(y_true=batch_gt_p3, y_pred=pred_p3)
                    loss_p4 = self._loss_fn(y_true=batch_gt_p4, y_pred=pred_p4)
                    loss_p5 = self._loss_fn(y_true=batch_gt_p5, y_pred=pred_p5)
                    loss_p6 = self._loss_fn(y_true=batch_gt_p6, y_pred=pred_p6)
                    loss_p7 = self._loss_fn(y_true=batch_gt_p7, y_pred=pred_p7)
                    total_loss = loss_p1 + loss_p2 + loss_p3 + loss_p4 + loss_p5 + loss_p6 + loss_p7

                grads = tape.gradient(total_loss, self._model.trainable_weights)
                self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

                pred_index_1 = tf.argmax(pred_p1, axis=-1)  # shape = (batch_size, )
                pred_index_2 = tf.argmax(pred_p2, axis=-1)  # shape = (batch_size, )
                pred_index_3 = tf.argmax(pred_p3, axis=-1)  # shape = (batch_size, )
                pred_index_4 = tf.argmax(pred_p4, axis=-1)  # shape = (batch_size, )
                pred_index_5 = tf.argmax(pred_p5, axis=-1)  # shape = (batch_size, )
                pred_index_6 = tf.argmax(pred_p6, axis=-1)  # shape = (batch_size, )
                pred_index_7 = tf.argmax(pred_p7, axis=-1)  # shape = (batch_size, )

                flag_index_1 = tf.math.equal(x=pred_index_1, y=batch_gt_p1)  # shape = (batch_size, )
                flag_index_2 = tf.math.equal(x=pred_index_2, y=batch_gt_p2)  # shape = (batch_size, )
                flag_index_3 = tf.math.equal(x=pred_index_3, y=batch_gt_p3)  # shape = (batch_size, )
                flag_index_4 = tf.math.equal(x=pred_index_4, y=batch_gt_p4)  # shape = (batch_size, )
                flag_index_5 = tf.math.equal(x=pred_index_5, y=batch_gt_p5)  # shape = (batch_size, )
                flag_index_6 = tf.math.equal(x=pred_index_6, y=batch_gt_p6)  # shape = (batch_size, )
                flag_index_7 = tf.math.equal(x=pred_index_7, y=batch_gt_p7)  # shape = (batch_size, )

                temp_a = tf.math.logical_and(x=flag_index_1, y=tf.math.logical_and(flag_index_2, flag_index_3))
                temp_b = tf.math.logical_and(x=flag_index_4, y=tf.math.logical_and(flag_index_5, flag_index_6))
                logic_and_res = tf.math.logical_and(x=temp_a, y=tf.math.logical_and(temp_b, flag_index_7))
                correct_records_amount = tf.reduce_sum(tf.cast(logic_and_res, tf.float32))
                recognition_accuracy = correct_records_amount / self._batch_size

                global_batch_counter += 1

                print("[Phase = Training] Epoch = {}, Batch = {} / {}: Loss = {}, Accuracy = {}".format(
                    epoch + 1, batch_idx + 1, self._batch_generator.batch_amount(phase='Train'),
                    float(total_loss), float(recognition_accuracy)))

                if global_batch_counter % self._save_model_every_batches == 0:
                    # evaluation over validation dataset
                    valid_overall_accuracy, valid_overall_loss = self.evaluate()
                    print("[Phase = Validation] Loss = {}, Accuracy = {}".format(
                        float(valid_overall_loss), float(valid_overall_accuracy)))

                    model_dump_root_dir = os.path.join(self._dump_models_root_dir, 'iter_{}'.format(global_batch_counter))
                    self._model.save(model_dump_root_dir)
                    print('Saving model into {}...'.format(model_dump_root_dir))


if __name__ == '__main__':
    license_plate_recognizer = LicensePlateRecognizer(
        dataset_root_dir='../IPCVehicleLicenseRoIForRecognition',
        dump_models_root_dir='./models',
        batch_size=128,
        save_model_every_batches=8000
    )

    license_plate_recognizer.train(max_epochs=50)
