

import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import ARM64, LINUX, LOGGER, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
from ultralytics.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):

        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))["names"]
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix      |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx (dnn=True)|
            | OpenVINO              | *openvino_model/ |
            | CoreML                | *.mlpackage      |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
            | NCNN                  | *_ncnn_model     |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolov8n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            ncnn,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton
        nhwc = coreml or saved_model or pb or tflite or edgetpu
        stride = 32
        model, metadata = None, None

        cuda = torch.cuda.is_available() and device.type != "cpu" 
        if cuda and not any([nn_module, pt, jit, engine, onnx]):
            device = torch.device("cpu")
            cuda = False

        if not (pt or triton or nn_module):
            w = attempt_download_asset(w)

        if nn_module:
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape
            stride = max(int(model.stride.max()), 32)
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model
            pt = True
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape
            stride = max(int(model.stride.max()), 32)
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model
        elif jit:
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))
        elif dnn:
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
        elif xml:
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.3")
            import openvino as ov

            core = ov.Core()
            w = Path(w)
            if not w.is_file():
                w = next(w.glob("*.xml"))
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))
            batch_dim = ov.get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")
            metadata = w.parent / "metadata.yaml"
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            try:
                import tensorrt as trt
            except ImportError:
                if LINUX:
                    check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
                import tensorrt as trt
            check_version(trt.__version__, "7.0.0", hard=True)
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)

            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]
        elif coreml:
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            metadata = dict(model.user_defined_metadata)
        elif saved_model:
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"
        elif pb:
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            from ultralytics.engine.exporter import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap frozen graphs for deployment."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
        elif tfjs:
            raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            w = Path(w)
            if not w.is_file():
                w = next(w.rglob("*.pdmodel"))
            config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w.parents[1] / "metadata.yaml"
        elif ncnn:
            LOGGER.info(f"Loading {w} for NCNN inference...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():
                w = next(w.glob("*.param"))
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. "
                "See https://docs.ultralytics.com/modes/predict for help."
                f"\n\n{export_formats()}"
            )

        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ("stride", "batch"):
                    metadata[k] = int(v)
                elif k in ("imgsz", "names", "kpt_shape") and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING 鈿狅笍 Metadata not found for 'model={weights}'")

        if "names" not in locals():
            names = default_class_names(data)
        names = check_class_names(names)

        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)

        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)
        elif self.jit:
            y = self.model(im)
        elif self.dnn:
            im = im.cpu().numpy()
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:
            im = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:
            im = im.cpu().numpy()
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))

            y = self.model.predict({"image": im_pil})
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )





            elif len(y) == 1:
                y = list(y.values())
            elif len(y) == 2:
                y = list(reversed(y.values()))
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            ex = self.net.create_extractor()
            input_names, output_names = self.net.input_names(), self.net.output_names()
            ex.input(input_names[0], mat_in)
            y = []
            for output_name in output_names:
                mat_out = self.pyncnn.Mat()
                ex.extract(output_name, mat_out)
                y.append(np.array(mat_out)[None])
        elif self.triton:
            im = im.cpu().numpy()
            y = self.model(im)
        else:
            im = im.cpu().numpy()
            if self.saved_model:
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:
                y = self.frozen_func(x=self.tf.constant(im))
                if len(y) == 2 and len(self.names) == 999:
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4
                    self.names = {i: f"class{i}" for i in range(nc)}
            else:
                details = self.input_details[0]
                integer = details["dtype"] in (np.int8, np.int16)
                if integer:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if integer:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale
                    if x.ndim > 2:


                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                    y.append(x)

            if len(y) == 2:
                if len(y[1].shape) != 4:
                    y = list(reversed(y))
                y[1] = np.transpose(y[1], (0, 3, 1, 2))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]


        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
            for _ in range(2 if self.jit else 1):
                self.forward(im)

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        This function takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml,
        engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

        Args:
            p: path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()
        """
        from ultralytics.engine.exporter import export_formats

        sf = list(export_formats().Suffix)
        if not is_url(p, check=False) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        types[8] &= not types[9]
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return types + [triton]

