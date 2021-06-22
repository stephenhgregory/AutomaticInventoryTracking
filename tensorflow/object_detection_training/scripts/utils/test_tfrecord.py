import tensorflow as tf 
import glob

# Get the raw dataset of TFRecords
filenames = glob.glob('data/*.tfrecord')
coco_1class_train_name = [i for i in filenames if 'coco_2017_train_1class.tfrecord' in i]
raw_dataset = tf.data.TFRecordDataset(coco_1class_train_name)
print(raw_dataset)

# Parse these serialized tensors using tf.train.Example.ParseFromString
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
