import tensorflow as tf

class BaseModel(object):
    def __init__(self, cfg, logger):
        self._cfg = cfg
        self._logger = logger
        self._loss_collection = 'losses'

        self._add_input_layer()
        self._add_forward_layer()
        self._add_output_layer()

        self._add_loss_layer()


    def _add_input_layer(self):
        raise NotImplementedError

    def _add_forward_layer(self):
        raise NotImplementedError

    def _add_output_layer(self):
        raise NotImplementedError

    def _add_loss_layer(self):
        raise NotImplementedError

    @property
    def all_variables(self):
        all_variables = tf.global_variables()
        return all_variables

    @property
    def bn_variables(self):
        all_variables = tf.global_variables()
        bn_variables = []
        for var in all_variables:
            var_name = var.op.name
            var_basename = var_name.split('/')[-1]
            if 'bn' in var_name and ('moving_mean' == var_basename or 'moving_variance' == var_basename):
                bn_variables.append(var)
        return bn_variables

    @property
    def bn_mean_variance(self):
        default_graph = tf.get_default_graph()
        all_nodes = default_graph.as_graph_def().node
        bn_tensors = []
        for node in all_nodes:
            node_name = node.name
            node_basename = node.name.split('/')[-1]
            if 'bn' in node_name and ('mean' == node_basename or 'variance' == node_basename):
                tensor = default_graph.get_tensor_by_name(node_name + ':0')
                bn_tensors.append(tensor)
        return bn_tensors

    @property
    def trainables(self):
        trainables = tf.trainable_variables()
        return trainables

    @property
    def initializer(self):
        initializer = tf.global_variables_initializer()
        return initializer

