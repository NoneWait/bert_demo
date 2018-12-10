import collections
import json
import random
import tensorflow as tf


class AICExample(object):
    def __init__(self,
                 qas_id,
                 quesion_text,
                 doc_text,
                 orig_answer,
                 alters,
                 labels):
        self.qas_id = qas_id
        self.quesion_text = quesion_text
        self.doc_text = doc_text
        self.orig_answer = orig_answer
        self.alters = alters
        self.labels = labels


def read_aic_examples(input_file, is_training):
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)

    examples = []
    for sample in input_data:
        qas_id = sample["query_id"]
        quesion_text = " ".join(sample["query"])
        doc_text = " ".join(sample["passage"])
        alters = sample["alternatives"]
        random.shuffle(alters)
        answer = sample["answer"]
        label = 0
        for index, alter in enumerate(alters):
            if alter.strip('\n') == answer.strip('\n'):
                label = index
        example = AICExample(
            qas_id=qas_id,
            quesion_text=quesion_text,
            doc_text=doc_text,
            orig_answer=answer,
            alters=alters,
            labels=label
        )
        examples.append(example)

    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 input_ids,
                 input_mask,
                 segment_ids,
                 labels):
        self.unique_id = unique_id
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                 is_training, output_fn):
    """Loads a data file into a list of 'InputBatch's"""
    unique_id = 1000000000
    for (example_index, example) in enumerate(examples):
        labels = example.labels
        query_tokens = tokenizer.tokenize(example.quesion_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        alter_tokens = []
        for alter in example.alters:
            alter_tokens += tokenizer.tokenize(alter)

        for token in alter_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # 在bert中输入是[CLS]sequence[SEP]sequence[SEP]...来设计的
        # [CLS],[SEP] [SEP] [SEP]
        # 3->for alter
        max_tokens_for_doc = max_seq_length - len(query_tokens) - len(alter_tokens) - 4

        doc_tokens = tokenizer.tokenize(example.doc_text)
        if len(doc_tokens) > max_tokens_for_doc:
            doc_tokens = doc_tokens[:max_tokens_for_doc]

        for token in doc_tokens:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append("[SEP]")
        segment_ids.append(2)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # 补齐
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=labels
        )
        # Run callback
        output_fn(feature)

        unique_id += 1


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {"unique_ids": tf.FixedLenFeature([], tf.int64),
                        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
                        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
                        "labels": tf.FixedLenFeature([], tf.int64)}

    # if is_training:

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        # if self.is_training:
        features["labels"] = create_int_feature([feature.labels])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()
