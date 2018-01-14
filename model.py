import tensorflow as tf

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
