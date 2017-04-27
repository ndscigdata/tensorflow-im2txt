# 谷歌图像叙事功能
基于tensorflow 1.0实现im2txt。也可见博客地址：[CSDN博客](http://blog.csdn.net/sparkexpert/article/details/70846094)

# 预训练模型下载
由于本人实验环境相对较差，没有GPU，所以没有测试训练过程。因此下载了个预训练模型。
下载地址如下所示：https://drive.google.com/file/d/0Bw6m_66JSYLlRFVKQ2tGcUJaWjA/view

### 运行环境介绍
* Python 3.6
* Tensorflow >= 1.0.1
* model/im2txt

# 测试过程中填“坑”
（1） word_counts.txt文件的处理，需要将文件中的 b' str'  ==>  str，即把字符串的引号等全部去掉。


（2）修改预训练模型中的名称，由于预训练模型的名称不一致的问题，所以需要进行修改。

在具体代码修改中，添加一个函数来进行模型的修改和重新保存

# 由于版本不同，需要进行修改
def RenameCkpt():
    vars_to_rename = {
    "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/weights",
    "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/biases",
    }
    new_checkpoint_vars = {}
    reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
    for old_name in reader.get_variable_to_shape_map():
      if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
      else:
        new_name = old_name
      new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)
    
    with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, "/home/ndscbigdata/work/change/tf/gan/im2txt/ckpt/newmodel.ckpt-2000000")
    print("checkpoint file rename successful... ")

# 训练结果：
	图片放在data目录下:
	![[image](./data/COCO_val2014_000000224477.jpg)]
图像 COCO_val2014_000000224477.jpg 标题是:
  0) a man riding a wave on top of a surfboard . (概率=0.035672)
  1) a person riding a surf board on a wave (概率=0.016238)
  2) a man on a surfboard riding a wave . (概率=0.010146)
  
	![[image](./data/ep271.jpg)]
图像 ep271.jpg 标题是:
  0) a woman is standing next to a horse . (概率=0.000759)
  1) a woman is standing next to a horse (概率=0.000647)
  2) a woman is standing next to a brown horse . (概率=0.000384)
  
	![[image](./data/dog.jpg)]
图像 dog.jpg 标题是:
  0) a dog is eating a slice of pizza . (概率=0.000138)
  1) a dog is eating a slice of pizza on a plate . (概率=0.000047)
  2) a dog is sitting at a table with a pizza on it . (概率=0.000039)

