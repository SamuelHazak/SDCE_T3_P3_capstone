from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import tensorflow as tf


# Based on Team Vulture's guide on how to train a Traffic Light Detector & Classifier with TF Object Detection API:
class TLClassifierSite(object):
    def __init__(self):
        # Handling cuDNN issues when using the model:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True

        # Prepare state list:
        self.states = [TrafficLight.UNKNOWN, TrafficLight.RED, TrafficLight.YELLOW,
                       TrafficLight.GREEN, TrafficLight.UNKNOWN, TrafficLight.UNKNOWN]

        # Load our frozen model:
        model_path = 'light_classification/models/frozen_graph_site.pb'
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as pbfile:
                # Read our frozen model and parse it:
                serial_graph = pbfile.read()
                graph_def.ParseFromString(serial_graph)
                tf.import_graph_def(graph_def, name='')

                # Start up the tensorflow session with our configuration
                self.sess = tf.Session(graph=self.graph, config=config)

                # Define tensor variables from the graph:
                self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')  # Input image tensor
                self.det_scores = self.graph.get_tensor_by_name('detection_scores:0')  # Scores tensor
                self.det_classes = self.graph.get_tensor_by_name('detection_classes:0')  # Classes tensor
                self.num_det = self.graph.get_tensor_by_name('num_detections:0')  # Number of predictions

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Preprocessing the image and expanding its dimensions before sending it to our net:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Images from bridge are BGR
        img = np.expand_dims(np.array(rgb_image), axis=0)

        # Run the session (prediction)
        with self.graph.as_default():
            (scores, classes, num) = self.sess.run([self.det_scores, self.det_classes, self.num_det],
                                                   feed_dict={self.image_tensor: img})
        # Dimension reduction:
        scores = np.squeeze(scores)  # Prediction scores (between 0 and 1)
        classes = np.squeeze(classes).astype(np.int32)  # Predictions

        # Check predictions:
        label_idx = 5  # UNKNOWN
        best_label_idx = classes[0]
        confidence_level = scores[0]
        classified_state = self.states[label_idx]

        if 0 < best_label_idx < 4 and confidence_level is not None and confidence_level > 0.5:
            label_idx = best_label_idx
            classified_state = self.states[label_idx]

        # print 'Prediction label =', label_idx
        # print 'Confidence level =', confidence_level

        return classified_state
