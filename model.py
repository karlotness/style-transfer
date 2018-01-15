# Copyright (c) 2018 Karl Otness
from collections import namedtuple
import tensorflow as tf

class StyleTransfer:
    TrainResult = namedtuple("TrainResult", ['total_loss', 'style_loss', 'content_loss'])

    def __init__(self, layers, art, image,
                 content_weight = 1E-4, style_weight = 1.0,
                 content_layer_weights = None, style_layer_weights = None,
                 learning_rate = 10, session = None):
        # Construct new session if needed
        self.session = session
        if self.session is None:
            self.session = tf.Session()
        with self.session.as_default(), tf.name_scope("styletransfer") as scope:
            # Prepare core VGG model
            self.vgg = Vgg(layers)
            # Create constants and variables for images
            self.art = tf.constant(art, 'float32', name = "art")
            self.image = tf.constant(image, 'float32', name = "image")
            gen_size = image.shape
            self.gen_image = tf.get_variable('gen_image', gen_size, dtype = 'float32',
                                             initializer = tf.random_uniform_initializer(minval=-20, maxval=20))
            # Copy in parameters into the class
            self.content_weight = content_weight
            self.style_weight = style_weight
            self.content_layer_weights = content_layer_weights
            # If no layer weights were specified, use defaults
            self.style_layer_weights = style_layer_weights
            if self.content_layer_weights is None:
                self.content_layer_weights = self.__get_default_content_weights()
            if self.style_layer_weights is None:
                self.style_layer_weights = self.__get_default_style_weights()
            self.learning_rate = learning_rate
            # Run the three images through VGG and gather activations
            self._art_activations = self.vgg.get_activations(self.art)
            self._image_activations = self.vgg.get_activations(self.image)
            self._gen_image_activations = self.vgg.get_activations(self.gen_image)
            # Compute the loss values
            self.style_loss = self.__get_style_loss()
            self.content_loss = self.__get_content_loss()
            with tf.name_scope("total_loss") as tot_scope:
                self.total_loss = self.style_weight * self.style_loss + self.content_weight * self.content_loss
            # Configure the optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize_op = self.optimizer.minimize(self.total_loss)

    def init(self):
        self.session.run(tf.global_variables_initializer())

    def train(self):
        _, total, style, content = self.session.run([self.optimize_op,
                                                     self.total_loss,
                                                     self.style_loss,
                                                     self.content_loss])
        return self.TrainResult(total, style, content)

    def get_image_data(self):
        return self.session.run(self.gen_image)

    @staticmethod
    def flatten_activation(activ):
        squeezed = tf.squeeze(activ)
        size = squeezed.shape[-1]
        return tf.reshape(squeezed, (-1, size))

    def __get_content_loss(self):
        with tf.name_scope("content_loss") as scope:
            loss = []
            for layer in range(len(self._image_activations)):
                c_wt = self.content_layer_weights[layer]
                if c_wt == 0:
                    continue
                gen_activ = self._gen_image_activations[layer]
                image_activ = self._image_activations[layer]
                diff = gen_activ - image_activ
                loss.append(c_wt * tf.reduce_sum(tf.pow(diff, 2)))
            return 0.5 * tf.reduce_sum(loss)

    def __get_style_loss(self):
        with tf.name_scope("style_loss") as scope:
            loss = []
            for layer in range(len(self._art_activations)):
                s_wt = self.style_layer_weights[layer]
                if s_wt == 0:
                    continue
                gen_activ = self._gen_image_activations[layer]
                art_activ = self._art_activations[layer]
                gen_flat = self.flatten_activation(gen_activ)
                art_flat = self.flatten_activation(art_activ)
                # Generate Gram matrices
                gen_gram_matrix = tf.matmul(tf.transpose(gen_flat), gen_flat,
                                            name = "gen_gram_matrix")
                art_gram_matrix = tf.matmul(tf.transpose(art_flat), art_flat,
                                            name = "art_gram_matrix")
                # Compute loss for this activation
                diff = gen_gram_matrix - art_gram_matrix
                sum_sq = tf.reduce_sum(tf.pow(diff, 2))
                # Compute scaling factors for this activation
                n = int(gen_activ.shape[3])
                m = int(gen_activ.shape[1]) * int(gen_activ.shape[2])
                mult = 1 / (4 * n**2 * m**2)
                loss.append(s_wt * mult * sum_sq)
            return tf.reduce_sum(loss)

    def __get_layer_nums(self):
        for l_name in self.vgg.layer_names:
            yield l_name[4:7]

    def __get_default_content_weights(self):
        c_weights = []
        for ln in self.__get_layer_nums():
            if ln == '4_1':
                c_weights.append(1)
            else:
                c_weights.append(0)
        return c_weights

    def __get_default_style_weights(self):
        count = sum(ln.endswith('_1') for ln in self.__get_layer_nums())
        s_weights = []
        for ln in self.__get_layer_nums():
            if ln.endswith('_1'):
                s_weights.append(1 / count)
            else:
                s_weights.append(0)
        return s_weights


class Vgg:
    def __init__(self, layers):
        self.layer_names = []
        self.__weights = []
        self.__biases = []
        current_group = None
        group_count = 0
        self.__layer_groups = []
        for layer in layers:
            self.layer_names.append(layer.name)
            weight = tf.constant(layer.weight, 'float32',
                                 name = layer.name+"_weight")
            bias = tf.constant(layer.bias, 'float32',
                               name = layer.name+"_bias")
            self.__weights.append(weight)
            self.__biases.append(bias)
            group = int(layer.name[4:].split('_')[0])
            if current_group is None:
                current_group = group
            if current_group == group:
                group_count += 1
            else:
                self.__layer_groups.append(group_count)
                current_group = group
                group_count = 1
        self.__layer_groups.append(group_count)

    def get_activations(self, input):
        activations = []
        layer_index = 0
        with tf.name_scope("vgg") as vgg_scope:
            expanded = tf.expand_dims(input, 0, name = "expand_input")
            activ = expanded
            for group_num, l_group in enumerate(self.__layer_groups):
                for i in range(l_group):
                    weight = self.__weights[layer_index]
                    bias = self.__biases[layer_index]
                    layer_name = self.layer_names[layer_index]
                    layer_index += 1
                    with tf.name_scope(layer_name) as scope:
                        conv_result = tf.nn.conv2d(activ, weight, [1, 1, 1, 1],
                                                   "SAME", name = "conv")
                        bias_result = tf.add(conv_result, bias, name = "bias")
                        activations.append(bias_result)
                        relu_result = tf.nn.relu(bias_result, name = "relu")
                        activ = relu_result
                activ = tf.nn.avg_pool(activ, [1, 2, 2, 1], [1, 2, 2, 1],
                                       "SAME",
                                       name = "pool_{}".format(group_num))
        return activations
