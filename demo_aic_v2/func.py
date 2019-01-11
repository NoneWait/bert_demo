import tensorflow as tf
INF = 1e30


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))
            # /sqrt(size) 使得内积不至于过大
            outputs = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def softmax_mask(val, mask):
    # 产生下溢出 =>使得e^(-inf) =>0
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dense(inputs, hidden, use_bias=True, reuse=False, scope="dense"):
    """
    :param inputs:
    :param hidden:
    :param use_bias:
    :param reuse:
    :param scope:
    :return: x*W+b
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))

            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args