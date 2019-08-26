from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf

# Handling cuDNN issues when using the model:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

global graph
graph = tf.get_default_graph()


class TLClassifierSim(object):
    def __init__(self):
        # Prepare state list:
        self.states = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN]

        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model('light_classification/models/model_sim_4.h5')

            # # Debug mode: print model layout:
            # for i, layer in enumerate(self.model.layers):
            #     print layer

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Prepare image for insertion to the model (which is based on MobileNet):
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image_resized = cv2.normalize(image_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_resized -= 0.5
        image_resized *= 2.
        img = np.expand_dims(np.array(image_resized), axis=0)

        with graph.as_default():
            # # Checking accuracy between all 4 labels:
            # proba = self.model.predict(img, batch_size=1, verbose=0)
            # print ("[DEBUG] Probablities = ", proba)

            # Get the prediction:
            label_idx = self.model.predict_classes(img, batch_size=1, verbose=0)
            # print "[DEBUG] Predicted label = %d" % label_idx[0]

        return self.states[label_idx[0]]
