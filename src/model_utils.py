# src/model_utils.py
import joblib
import os
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def save_pipeline(pipeline: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path: str):
    return joblib.load(path)

def export_pipeline_to_onnx(pipeline: Pipeline, sample_input, onnx_path: str):
    # sample_input should be a numpy array with shape (n_samples, n_features)
    initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)
    with open(onnx_path, 'wb') as f:
        f.write(onx.SerializeToString())