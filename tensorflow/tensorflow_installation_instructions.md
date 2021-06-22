# TensorFlow installation

### 1. Install TensorFlow
1. If you aren't already using the conda environment ```MBUSIBarcodeProject```, activate that now with this command
from a *Terminal* window
   ```
   pip install --ignore-installed --upgrade tensorflow==2.2.0
   ```
2. Verify your installation with the following command in a *Terminal* window
   ```
   python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
   ```
3. After the above command is run, you should see some output to the *Terminal* that looks similar to the one below:
   ```
    2020-06-22 19:20:32.614181: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2020-06-22 19:20:32.620571: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2020-06-22 19:20:35.027232: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
    2020-06-22 19:20:35.060549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
    pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1
    coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 238.66GiB/s
    2020-06-22 19:20:35.074967: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2020-06-22 19:20:35.084458: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
    2020-06-22 19:20:35.094112: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
    2020-06-22 19:20:35.103571: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
    2020-06-22 19:20:35.113102: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
    2020-06-22 19:20:35.123242: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
    2020-06-22 19:20:35.140987: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
    2020-06-22 19:20:35.146285: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2020-06-22 19:20:35.162173: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
    2020-06-22 19:20:35.178588: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x15140db6390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-06-22 19:20:35.185082: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-06-22 19:20:35.191117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-06-22 19:20:35.196815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]
    tf.Tensor(1620.5817, shape=(), dtype=float32)
   ```
   
## 2. Using GPU
For GPU support, view 
[here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#gpu-support-optional).

## 2. Download the Tensorflow Model Garden
1. Open a *Terminal* window in this directory.
    ```
    cd <insert_your_path>/CS-Fall-2020-Team-2/tensorflow
    ```
2. Then, clone the [TensorFlow Models repository](https://github.com/tensorflow/models) into this folder.
    ```
    git clone https://github.com/tensorflow/models.git
    ```
3. Now, you should have a folder named ```models``` inside of the ```CS-Fall-2020-Team-2/tensorflow/``` directory (this 
directory)
    
## 3. Install/Compile Protobuf
The TensorFlow Object Detection API uses something called Protobufs for configuration of models and hyperparameters.
Before we use the framework, the Protobuf libraries need to be downloaded and compiled.
Here's how:
1. Go to [protoc releases page](https://github.com/protocolbuffers/protobuf/releases)

2. Download the latest ```protoc-*-*.zip``` release

3. Extract the contents of your newly downloaded ```protoc-*-*.zip``` into a folder of your choice ```<PATH_TO_PB>```

4. Add ```<PATH_TO_PB>``` to your PATH environment variable
    - For information on how to add a path to the PATH variable in Linux, 
    [see this post](https://www.baeldung.com/linux/path-variable) (or many other places on the internet)
    
5. In a new *Terminal* window, move to the ```tensorflow/models/research``` repository with the following command
    ```
    # tensorflow/models should be in CS-Fall-2020-Team-2
    cd <insert_your_path>/CS-Fall-2020-Team-2/tensorflow/models/research
    ```
   
6. Execute the following command to compile protocol buffers
    ```
    protoc object_detection/protos/*.proto --python_out=.
    ```
   
## 4. Install the Object Detection API
Installation of the Object Detection API is done by installing the ```object_detection``` package.
This can easily be done by running the following commands from the same directory as the previous step, ```tensorflow/models/research```:
```
# From within tensorflow/models/research
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## 5. Test Installation
Run the following command from within ```<insert_your_path>/CS-Fall-2020-Team-2/tensorflow/models/research```
```
# From within tensorflow/models/research
python object_detection/builders/model_builder_tf2_test.py
```
This may take a moment, but the output should look something like this at the end:
```
...
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 68.510s

OK (skipped=1)
```

## 6. All done!
If you followed all of the steps properly, you should be setup!

**Note:** If the above steps gave you some other errors, check out 
[here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) for official 
Tensorflow documentation of the same exact installation that we just performed.