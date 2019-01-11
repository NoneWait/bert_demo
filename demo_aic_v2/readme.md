## 本流程根据官方Bert代码编写
## 代码主要分为三个部分：
- Example
    - 负责读取数据
    - 对数据转为特征
    - 将特征转为tf.record文件
- Model
    - 定义模型
    - 编写模型构造器，其中包含训练、验证、预测
- Run
    - 读取参数
    - 项目主入口

## 数据预处理
### Example
- 编写一个样本类
    - 注意点：(class->instance)类的方法的第一个参数永远是self，self指向创建的实例本身
    - 访问限制：\_\_name，属性名称前加两个下划线，则是私有变量(可以访问，但是不同python编译器会改成不同名字，比如 \_classname_name)
    - 默认方法：\_\_str\_\_会被print调用，\_\_repr\_\_会被控制台输出时默认调用
```python
class AICExample(object):
    def __init__(self,qas_id):
        self.qas_id = qas_id
    
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        s = self.qas_id
        return s
```

- 编写一个读取数据原文件，处理数据，将数据转为一个个样本类

```python
    def read_aic_example(input_file, is_training):
        ...
```

- 定义一个特征类，每个样本什么样的特征，然后将一个样本转为一个特征类
    - 其中，bert转换为特征向量是时候，分别要定义三个特征向量，input_ids,input_mask,segment_ids,
        - token：将两个句子拼接->'[CLS]'+sent1_tokens+'[SEP]'+sent2_tokens+'[SEP]'
        - input_ids:利用bert自带的tokenizer转为id
        - input_mask:有词的地方mask为1
        - segment_ids:[000111]。第一个句子mask为0,第二个句子mask为1
        - 多选选择(一个样本有多组句子对->如多项选择题)的话则处理成多个input_ids

```python
    class InputFeatures(object):
        def __init__(self, ...,labels):
            ...
            self.labels = labels
```

```python
    def convert_examples_to_features(...):
        ...
        # Run callback
        # 回调函数
        output_fn(feature)
```

- 定义一个特征写入器，作为上述回调函数，将每个特征类实例写入tf.record文件
 - tf.python_io.TFRecordWriter
 - Example定义了一种样本标准：一个样本由若干个feature组成，每个feature由key和value组成，很好理解
 - tf.train.Feature()
 - tf.train.Features()
 - tf.train.Example
```python
    FeatureWriter(object):
        def __init__(self, filename, is_training):
            ...
            self.num_features=0
            self._writer = tf.python_io.TFRecordWriter(filename)
        def process_feature(self, feature)       :
            self.num_feature += 1
            def create_int_feature(values):
                ...定义一个特征
            
            features['key'] = create_int_features('value')
            
            # 定义一个样本，包含若干个特征，key-value
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            
            # 将样本序列化（压缩）保存到tf.record文件中
            self.__writer.write(tf_example.SerializeToString())
        def close(self):
            #记得关文件
            self._writer.close()
```

- 设计一个模型的输出函数，完成读取tf.record文件，反序列化样本获得原始的样本，如果是训练的话，则打乱数据集，获取batch量的样本集
```
    def input_fn_builder(input_file, ..., drop_remainder)
        # 定义好存储在record文件里特征的key和size
        name_to_features={
            "input_ids": tf.FixedLenFeature([3, seq_length], tf.int64),
        }
        # 一个下划线代表是可以使用，但是最好不用的private function
        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            ... 这时example其实是一个字典,{feature _key:feature_value}
        def input_fn(params):
            batch_size = params["batch_size"]
            
            d = tf.data.TFRecordDataset(input_file)
```

## 模型定义
- 和编写自定义的Estimator一样，需要自定义模型，和编写模型函数（为三种模式指定不同计算）

- 建立模型
    - 需要注意的是：如果每个样本有多个input_ids输入，可以将原有的输入[batch,multi_size, seq_len]转为[batch\*multi_size, seq_len]，这样相当用同一个bert模型处理multi_size\*batch次的单输入
    - 根据不同需要，可以调用model返回的编码结果（pool后或者每层的encode）

```python 
    def create_model():
        model = modeling.BertModel(
        ...
        input_ids=tf.reshape(input_ids, [-1, input_ids.shape[-1].value]),
        ...
        )
```
- 模型函数
    - 构建模型函数的时候需要完成model_fh(features,labels,mode,params)这个函数
    - 这个函数需要提供代码来处理三种mode(TRAIN,EVAL,PREDICT)值，返回tf.estimator.EstimatorSpec的一个实例
    - train模式：主要是需要返回loss，train_op(优化器)
    - eval模式：主要是需要返回loss,eval_metrics=[评价函数]
    - predict模式：主要是需要返回predictions结果
```
def def model_fn_builder(bert_config, num_labels, init_checkpoint,...):
    def model_fn(features,labels,mode,params):
        ...=create_model(...)
        ...是否从checkpoint恢复
        ...计算梯度...
        if mode == tf.estimator.ModeKeys.TRAIN:
            ...定义优化器..
            output_spec = tf.estimator.EstimatorSpec(mode=mode,loss=损失,train_op=优化器)
        
        
```

- init_checkpoint，从checkpoint返回结果

```
# 其中assaigment_map={"old var names":"new var names",...}
tf.train.init_from_checkpoint(init_checkpoint, assigment_map)
```
```
# 获取所有变量名
tvars = tf.trainable_variables()
for var in tvars:
    print(var.name)
    print(var.shape)
```

## Run model
- 这部分主要是需要获取自定义参数，和构建运行逻辑

- 自定义参数：
```
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name, default='', description)

# 设定某个参数是必须给定的
flags.mark_flag_as_required(name)

```

- 预加载bert

```
# config
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
# tokenizer
# bert 自带的tokenizer
tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
```

- 配置run_config
    - model_dir：配置输出文件夹
    - save_checkpoint_steps:训练多少步保存checkpoint
    
- 加载数据集，计算训练步

``` 
# 这里不同的是，考虑了epoch
num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
```

- 构建模型和Estimator
    -  构建model_fn_builder()
    -  构建estimator
    ```
    tf.train.Estimator(model_fn=model_fn,config=run_config,train/eval/predict_batch)
    ```

- 训练/验证/预测
    - 构建input_fn_builder():指定如何解析样本（如从record文件中读取解析）
    - estimator.train(input_fn=...,max_steps=num_train_steps)，这个操作会将mode设为ModeKeys.TRAIN
    - predict，一个一个样例返回
    - result中包含了在model_fn中predict模式返回的实例中predictions参数的内容
    ```
    for result in estimator.predict(predict_input_fn,yield_single_examples=True):
        print(result)
    ```