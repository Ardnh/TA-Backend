import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

app = FastAPI()
# serving original image
app.mount("/original_img", StaticFiles(directory="original_img"), name="original_img")
# serving predicted image
app.mount("/predicted_img", StaticFiles(directory="predicted_img"), name="predicted_img")

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('./trained_model/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('./trained_model/ckpt-7').expect_partial()

category_index = label_map_util.create_category_index_from_labelmap('./trained_model/label_map.pbtxt')

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def save_picture(file):
    # randon_uid = str(uuid4())
    _, f_ext = os.path.splitext(file.filename)
    
    # picture_name = (randon_uid if fileName==None else fileName.lower().replace(' ', '')) + f_ext 
        
    picture_path = os.path.join("original_img",file.filename)

    img = Image.open(file.file)
    img.save(picture_path)
    
    return picture_path
 
@app.post("/predict")
async def predict(file: UploadFile):

    # read image from outside
    image_np = np.array(tf.image.decode_jpeg(file.file.read(), channels=3))
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # save original image
    file_path = save_picture(file)

    # predict image
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                max_boxes_to_draw=1,
                min_score_thresh=.5,
                agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2RGBA))

    plt.axis('off')
    plt.savefig(os.path.join('./predicted_img/', file.filename), bbox_inches='tight', pad_inches=0)

    predictedImagePath = os.path.join('/predicted_img', file.filename)

    return {
        "status": "OK",
        "predicted_image_path": predictedImagePath, 
        "original_image_path": file_path,
    }
