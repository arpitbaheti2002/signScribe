def train_model():
    import os
    import wget

    subprocess.run(['pip', 'install', 'tensorflow==2.4.1'])
    subprocess.run(['pip', 'install', 'keras==2.3.1'])
    subprocess.run(['pip', 'install', 'numpy==1.23.4'])

    import shutil
    try:
        shutil.rmtree(os.path.join('Tensorflow', 'workspace', 'annotations'))
    except: 
        pass

    try:
        shutil.rmtree(os.path.join('Tensorflow', 'workspace', 'models'))
    except: 
        pass

    try:
        shutil.rmtree(os.path.join('Tensorflow', 'workspace', 'pre-trained-models'))
    except: 
        pass

    CUSTOM_MODEL_NAME = 'my_ssd_mobilenet-new' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
    }

    ACTIONS_RECORD = os.path.join('Tensorflow', 'workspace', 'actions.txt')

    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    for path in paths.values():
        if not os.path.exists(path):
            if os.name == 'posix':
                os.system('mkdir -p {path}')
            if os.name == 'nt':
                os.system('!mkdir {path}')

    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        os.system('git clone https://github.com/tensorflow/models {paths["APIMODEL_PATH"]}')

    if os.name=='posix':  
        os.system('apt-get install protobuf-compiler')
        os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .') 
        
    if os.name=='nt':
        url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        os.system('move protoc-3.15.6-win64.zip {paths["PROTOC_PATH"]}')
        os.system('cd {paths["PROTOC_PATH"]} && tar -xf protoc-3.15.6-win64.zip')
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
        os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install')
        os.system('cd Tensorflow/models/research/slim && pip install -e . ')

    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'



    if os.name =='posix':
        os.system('wget {PRETRAINED_MODEL_URL}')
        os.system('mv {PRETRAINED_MODEL_NAME+".tar.gz"} {paths["PRETRAINED_MODEL_PATH"]}')
        os.system('cd {paths["PRETRAINED_MODEL_PATH"]} && tar -zxvf {PRETRAINED_MODEL_NAME+".tar.gz"}')
    if os.name == 'nt':
        wget.download(PRETRAINED_MODEL_URL)
        os.system('move {PRETRAINED_MODEL_NAME+".tar.gz"} {paths["PRETRAINED_MODEL_PATH"]}')
        os.system('cd {paths["PRETRAINED_MODEL_PATH"]} && tar -zxvf {PRETRAINED_MODEL_NAME+".tar.gz"}')

    import object_detection

    actions = []
    labels = []
    with open(ACTIONS_RECORD, 'r') as file:
        actions = file.read().splitlines()

    id = 1
    for action in actions:
        labels.append({'name': action, 'id': id})
        id += 1

    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    if os.path.exists(ARCHIVE_FILES):
        os.system('tar -zxvf {ARCHIVE_FILES}')

    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        os.system('git clone https://github.com/nicknochnack/GenerateTFRecord {paths["SCRIPTS_PATH"]}')

    os.system('python {files["TF_RECORD_SCRIPT"]} -x {os.path.join(paths["IMAGE_PATH"], "collectedimages")} -l {files["LABELMAP"]} -o {os.path.join(paths["ANNOTATION_PATH"], "train.record")}')

    if os.name =='posix':
        os.system('cp {os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "pipeline.config")} {os.path.join(paths["CHECKPOINT_PATH"])}')
    if os.name == 'nt':
        os.system('copy {os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "pipeline.config")} {os.path.join(paths["CHECKPOINT_PATH"])}')

    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format

    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)  

    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2500".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
    print(command)

    subprocess.run(['pip', 'install', 'tensorflow==2.9.0'])
    

    os.system(command)

    return 'Training Successful.'