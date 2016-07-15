import tensorflow as tf

import image_test_input
import numpy as np


class ImageInputTest(tf.test.TestCase):

    def testSimple(self):
        with self.test_session() as sess:
            q = tf.FIFOQueue(99, [tf.string], shapes=())
            q.enqueue(['/home/ttf/image_data_test/image_test.bin']).run()
            result = image_test_input.read_record(q)

            for i in range(128):
                key, label, image_name_1, image_name_2, ims = sess.run([
                    result.key, result.label, result.image_name_1,
                    result.image_name_2, result.uint8image
                ])
                print('label is {}, image_name_1 is {}, image_name_2 is {},'
                      'image shape is {}'.format(label, image_name_1.tostring(), image_name_2.tostring(),
                                                 ims.shape)
                      )

                images, labels, names_1, names_2 = image_test_input.inputs(eval_data='123')
                tf.train.start_queue_runners(sess=sess)
                images_new, labels_new, names_1_new, names_2_new = \
                    sess.run([images, labels, names_1, names_2])

                for image, label, name_1, name_2 in zip(images_new, labels_new, names_1_new, names_2_new):
                    print image.shape, label, name_1, name_2



if __name__ == '__main__':
    tf.test.main()