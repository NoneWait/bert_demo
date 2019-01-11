# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import collections
import numpy as np
import json

import tensorflow as tf


import modeling
import tokenization
from demo_aic_v2.AICExample import read_aic_examples, FeatureWriter, input_fn_builder, convert_examples_to_features
from demo_aic_v2.model import model_fn_builder

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
## checkpoint
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run predict on the predict set.")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    # bert 自带的tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        # 读取样本
        train_examples = read_aic_examples(
            input_file=FLAGS.train_file, is_training=True)
        # for test
        # train_examples = train_examples[:20]
        # print(len(train_examples))
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    # create model
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        num_labels=3)
    # If TPU is not available, this will fall back to normal Estimator on CPU
    #  or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.predict_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        # 利用tfrecord存储
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = read_aic_examples(input_file=FLAGS.predict_file, is_training=False)
        # for test
        # eval_examples = eval_examples[:3]
        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        # all_results = []

        eval_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = read_aic_examples(input_file=FLAGS.predict_file, is_training=False)
        # for test
        # predict_examples = predict_examples[:1]
        predict_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "predict.tf_record"),
            is_training=False)
        predict_features = []

        # This tells the estimator to run through the entire set.
        predict_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            predict_steps = int(len(predict_examples) / FLAGS.predict_batch_size)

        def append_feature(feature):
            predict_features.append(feature)
            predict_writer.process_feature(feature)

        convert_examples_to_features(
            examples=predict_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False,
            output_fn=append_feature)
        predict_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(predict_examples))
        tf.logging.info("  Num split examples = %d", len(predict_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        # all_results = []

        predict_input_fn = input_fn_builder(
            input_file=predict_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            logits = [float(x) for x in result["logits"].flat]
            probabilities = [float(x) for x in result["probabilities"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    logits=logits,
                    probabilities=probabilities))

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")

        write_predictions(predict_examples, predict_features, all_results, output_prediction_file)


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "logits", "probabilities"])

def write_predictions(predict_examples, predict_features, all_results, output_prediction_file):
    # print("examples")
    # print(predict_examples)
    # print("result:")
    # print(all_results)
    results = []
    for i, example in enumerate(predict_examples):
        orig_answer = example.orig_answer
        predict_label = np.argmax(all_results[i].probabilities)
        predict_answer = example.alters[predict_label]
        results.append({"doc": "".join(example.doc_text.split(" ")),
                        "ques": "".join(example.question_text.split(" ")),
                        "predict": predict_answer,
                        "orig_answer": orig_answer,
                        "alters": example.alters
                        })

    with open(output_prediction_file, encoding="utf-8", mode="w") as fh:
        json.dump(results, fh, ensure_ascii=False)

    # pass

if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
