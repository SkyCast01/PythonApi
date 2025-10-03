# from typing import List, Dict, Optional, Any
# from io import BytesIO
# from pathlib import Path
# import os
# import re
# import tempfile
# import zipfile
# import fnmatch
# import numpy as np
# import joblib
# from fastapi import FastAPI, HTTPException, File, UploadFile, Body
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from deep_translator import GoogleTranslator
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# import nibabel as nib
# from scipy import ndimage
# from PIL import Image

# # ---- TFLite interpreter shim (prefer LiteRT, then tflite-runtime, then TensorFlow) ----
# try:
#     from ai_edge_litert.interpreter import Interpreter  # up-to-date runtime
#     _TFLITE_SOURCE = "ai-edge-litert"
# except Exception:
#     try:
#         import tflite_runtime.interpreter as _tflite
#         Interpreter = _tflite.Interpreter
#         _TFLITE_SOURCE = "tflite-runtime"
#     except Exception:
#         import tensorflow as tf  # type: ignore
#         Interpreter = tf.lite.Interpreter  # type: ignore
#         _TFLITE_SOURCE = "tensorflow"

# # -----------------------------
# # Config
# # -----------------------------
# MODEL_DIR = Path(os.getenv("MODEL_DIR", "./AiModels")).resolve()
# SYMPTOM_PREDICTOR_PATH = MODEL_DIR / "disease_predictor.pkl"
# SYMPTOM_LABEL_PATH = MODEL_DIR / "label_encoder.pkl"
# SYMPTOM_LIST_PATH = MODEL_DIR / "symptom_list.pkl"
# BRAIN_MODEL_PATH = MODEL_DIR / "ResNet50V2.tflite"  # TFLite model
# HEART_MODEL_PATH = MODEL_DIR / "xgb_balanced.pkl"
# HEART_SCALER_PATH = MODEL_DIR / "scaler_balanced.pkl"
# MOOD_VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
# MOOD_MODEL_PATH = MODEL_DIR / "naivebayes_Model.pkl"
# AMARZERO_MODEL_PATH = Path(os.getenv("AMARZERO_MODEL_PATH", MODEL_DIR / "AmarZero_Medical_System.pth")).resolve()
# BONE_MODEL_PATH = MODEL_DIR / "bone_fracture_resnet18.pth"

# app = FastAPI(title="Medical Prediction Service", version="7.7.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt", quiet=True)

# try:
#     nltk.data.find("corpora/stopwords")
# except LookupError:
#     nltk.download("stopwords", quiet=True)

# # -----------------------------
# # Schemas
# # -----------------------------
# class BoneFractureResponse(BaseModel):
#     predicted_class: str
#     probabilities: Dict[str, float]

# class SymptomRequest(BaseModel):
#     symptoms: List[str]

# class TopProbItem(BaseModel):
#     disease: str
#     percent: str

# class PredictResponse(BaseModel):
#     predicted_disease: str
#     top_three: List[TopProbItem]

# class HeartDiseaseInput(BaseModel):
#     gender_male: int
#     age_years: float
#     smoker_current: int
#     on_bp_meds: int
#     has_diabetes: int
#     cholesterol_total: float
#     bp_systolic: float
#     heart_rate_bpm: float

# class HeartDiseaseResponse(BaseModel):
#     result: str
#     probability: float
#     prediction: int

# class MoodRequest(BaseModel):
#     text: str

# class MoodResponse(BaseModel):
#     mood: str
#     language: str
#     cleaned_text: Optional[str] = None
#     translated_text: Optional[str] = None
#     error: Optional[str] = None

# class AmarZeroResponse(BaseModel):
#     patient_folder: Optional[str]
#     files_present: List[str]
#     tumor_positions_count: int
#     stats: Optional[str] = None
#     error: Optional[str] = None

# # -----------------------------
# # Global Variables
# # -----------------------------
# SYMPTOM_MODEL = None
# SYMPTOM_LABEL = None
# SYMPTOM_LIST: List[str] = []
# NORM_TO_INDEX: Dict[str, int] = {}
# load_error_symptom: Optional[Exception] = None

# # TFLite brain model globals
# BRAIN_INTERPRETER: Optional[Any] = None
# BRAIN_INPUT_INDEX: Optional[int] = None
# BRAIN_OUTPUT_INDEX: Optional[int] = None
# # Quantization metadata (if any)
# BRAIN_IN_DTYPE: Optional[np.dtype] = None
# BRAIN_IN_SCALE: Optional[float] = None
# BRAIN_IN_ZP: Optional[int] = None
# BRAIN_OUT_DTYPE: Optional[np.dtype] = None
# BRAIN_OUT_SCALE: Optional[float] = None
# BRAIN_OUT_ZP: Optional[int] = None

# class_labels: List[str] = []
# load_error_brain: Optional[Exception] = None

# HEART_MODEL = None
# HEART_SCALER = None
# load_error_heart: Optional[Exception] = None

# MOOD_VECTORIZER = None
# MOOD_MODEL = None
# load_error_mood: Optional[Exception] = None

# AMARZERO_CHECKPOINT = None
# load_error_amarzero: Optional[Exception] = None

# BONE_MODEL = None
# BONE_TRANSFORM = None
# BONE_CLASSES = ["Fractured", "NO Fractured"]
# load_error_bone: Optional[Exception] = None

# # -----------------------------
# # Normalization / Synonyms
# # -----------------------------
# def _normalize_name(s: str) -> str:
#     if s is None:
#         return ""
#     s = s.strip().lower()
#     s = s.replace("-", "_")
#     s = re.sub(r"\s+", "_", s)
#     s = re.sub(r"[^a-z0-9_]+", "", s)
#     s = re.sub(r"_+", "_", s).strip("_")
#     s = s.replace("spotting__urination", "spotting_urination")
#     s = s.replace("dischromic__patches", "dischromic_patches")
#     return s

# SYNONYMS: Dict[str, List[str]] = {
#     "shortness_of_breath": ["breathlessness"],
#     "sob": ["breathlessness"],
#     "fever": ["high_fever", "mild_fever"],
#     "tiredness": ["fatigue"],
#     "stomach_pain": ["stomach_pain", "abdominal_pain", "belly_pain"],
#     "abdominal_pain": ["abdominal_pain", "stomach_pain", "belly_pain"],
#     "belly_pain": ["belly_pain", "abdominal_pain", "stomach_pain"],
#     "diarrhea": ["diarrhoea"],
#     "runny_nose": ["runny_nose"],
#     "foul_smell_of_urine": ["foul_smell_of_urine"],
#     "spotting_urination": ["spotting_urination"],
#     "dischromic_patches": ["dischromic_patches"],
#     "chest_pain": ["chest_pain"],
# }

# # -----------------------------
# # Translation helpers
# # -----------------------------
# def detect_language(text: str) -> str:
#     if text and re.search(r"[\u0600-\u06FF]", text):
#         return "ar"
#     return "en"

# def try_translate_to_english(text: str) -> Optional[str]:
#     try:
#         translated = GoogleTranslator(source="ar", target="en").translate(text)
#         return translated
#     except Exception as e:
#         print("Translation failed:", e)
#         return None

# # -----------------------------
# # Text cleaning
# # -----------------------------
# def clean_text(text: str, lang: str = "en") -> str:
#     text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
#     text = re.sub(r"@\w+", "", text)
#     text = re.sub(r"#\w+", "", text)
#     text = re.sub(r"[^\w\s]", "", text)
#     text = text.lower().strip()

#     sw = set()
#     try:
#         if lang == "ar":
#             sw = set(stopwords.words("arabic"))
#         else:
#             sw = set(stopwords.words("english"))
#     except Exception:
#         sw = set()

#     try:
#         tokens = word_tokenize(text)
#     except Exception:
#         tokens = text.split()

#     text = " ".join([w for w in tokens if w and w not in sw])
#     return text

# # -----------------------------
# # Loaders
# # -----------------------------
# def load_bone_model():
#     global BONE_MODEL, BONE_TRANSFORM, load_error_bone
#     try:
#         model = models.resnet18(pretrained=False)
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 2)
#         model.load_state_dict(torch.load(BONE_MODEL_PATH, map_location=torch.device("cpu")))
#         model.eval()
#         BONE_MODEL = model
#         BONE_TRANSFORM = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         load_error_bone = None
#         print("✅ Bone fracture ResNet18 loaded")
#     except Exception as e:
#         load_error_bone = e
#         print(f"❌ Bone fracture model error: {e}")

# def load_symptom_stack():
#     global SYMPTOM_MODEL, SYMPTOM_LABEL, SYMPTOM_LIST, NORM_TO_INDEX, load_error_symptom
#     try:
#         SYMPTOM_MODEL = joblib.load(SYMPTOM_PREDICTOR_PATH)
#         SYMPTOM_LABEL = joblib.load(SYMPTOM_LABEL_PATH)
#         if not SYMPTOM_LIST_PATH.exists():
#             raise FileNotFoundError(f"Missing: {SYMPTOM_LIST_PATH}")
#         loaded = joblib.load(SYMPTOM_LIST_PATH)
#         if not isinstance(loaded, (list, tuple)) or not all(isinstance(s, str) for s in loaded):
#             raise ValueError("symptom_list.pkl must be a list[str] in training order.")
#         SYMPTOM_LIST = list(loaded)
#         n_expected = getattr(SYMPTOM_MODEL, "n_features_in_", None)
#         if not isinstance(n_expected, (int, np.integer)):
#             raise ValueError("Model missing n_features_in_.")
#         if len(SYMPTOM_LIST) != int(n_expected):
#             raise ValueError(
#                 f"Training symptom list length ({len(SYMPTOM_LIST)}) "
#                 f"!= model.n_features_in_ ({n_expected}). Fix symptom_list.pkl."
#             )
#         NORM_TO_INDEX.clear()
#         for i, raw in enumerate(SYMPTOM_LIST):
#             norm = _normalize_name(raw)
#             NORM_TO_INDEX[norm] = i
#         load_error_symptom = None
#         print(f"✅ Symptom stack ready: {len(SYMPTOM_LIST)} features (exact match).")
#     except Exception as e:
#         load_error_symptom = e
#         print(f"❌ Symptom stack error: {e}")

# def load_brain_model():
#     """Load TFLite model, capture input/output details, and prepare labels."""
#     global BRAIN_INTERPRETER, BRAIN_INPUT_INDEX, BRAIN_OUTPUT_INDEX
#     global BRAIN_IN_DTYPE, BRAIN_IN_SCALE, BRAIN_IN_ZP
#     global BRAIN_OUT_DTYPE, BRAIN_OUT_SCALE, BRAIN_OUT_ZP
#     global class_labels, load_error_brain
#     try:
#         BRAIN_INTERPRETER = Interpreter(model_path=str(BRAIN_MODEL_PATH))
#         BRAIN_INTERPRETER.allocate_tensors()

#         in_details = BRAIN_INTERPRETER.get_input_details()[0]
#         out_details = BRAIN_INTERPRETER.get_output_details()[0]
#         BRAIN_INPUT_INDEX = in_details["index"]
#         BRAIN_OUTPUT_INDEX = out_details["index"]

#         # Cache dtypes and quantization metadata (if any)
#         BRAIN_IN_DTYPE = in_details.get("dtype", np.float32)
#         q_in = in_details.get("quantization_parameters", {}) or in_details.get("quantization", None)
#         if isinstance(q_in, dict):
#             scales = q_in.get("scales", [])
#             zero_points = q_in.get("zero_points", [])
#             BRAIN_IN_SCALE = float(scales[0]) if len(scales) else None
#             BRAIN_IN_ZP = int(zero_points[0]) if len(zero_points) else None
#         else:
#             # Old API: quantization is a tuple (scale, zero_point)
#             if isinstance(q_in, tuple) and len(q_in) == 2:
#                 BRAIN_IN_SCALE = float(q_in[0]) or None
#                 BRAIN_IN_ZP = int(q_in[1]) if q_in[1] is not None else None
#             else:
#                 BRAIN_IN_SCALE = None
#                 BRAIN_IN_ZP = None

#         BRAIN_OUT_DTYPE = out_details.get("dtype", np.float32)
#         q_out = out_details.get("quantization_parameters", {}) or out_details.get("quantization", None)
#         if isinstance(q_out, dict):
#             scales = q_out.get("scales", [])
#             zero_points = q_out.get("zero_points", [])
#             BRAIN_OUT_SCALE = float(scales[0]) if len(scales) else None
#             BRAIN_OUT_ZP = int(zero_points[0]) if len(zero_points) else None
#         else:
#             if isinstance(q_out, tuple) and len(q_out) == 2:
#                 BRAIN_OUT_SCALE = float(q_out[0]) or None
#                 BRAIN_OUT_ZP = int(q_out[1]) if q_out[1] is not None else None
#             else:
#                 BRAIN_OUT_SCALE = None
#                 BRAIN_OUT_ZP = None

#         # Determine number of classes from output shape
#         out_shape = out_details.get("shape", None)
#         num_classes = int(out_shape[-1]) if (out_shape is not None and len(out_shape) >= 1) else 4
#         class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary'] if num_classes == 4 \
#                        else [f"Class_{i}" for i in range(num_classes)]
#         load_error_brain = None
#         print(f"✅ Brain TFLite model loaded via {_TFLITE_SOURCE} with labels: {class_labels}")
#     except Exception as e:
#         load_error_brain = e
#         print(f"❌ Brain model error: {e}")

# def load_heart_model():
#     global HEART_MODEL, HEART_SCALER, load_error_heart
#     try:
#         HEART_MODEL = joblib.load(HEART_MODEL_PATH)
#         HEART_SCALER = joblib.load(HEART_SCALER_PATH)
#         load_error_heart = None
#         print("✅ Heart model + scaler loaded")
#     except Exception as e:
#         load_error_heart = e
#         print(f"❌ Heart model error: {e}")

# def load_mood_stack():
#     global MOOD_VECTORIZER, MOOD_MODEL, load_error_mood
#     try:
#         MOOD_VECTORIZER = joblib.load(MOOD_VECTORIZER_PATH)
#         MOOD_MODEL = joblib.load(MOOD_MODEL_PATH)
#         load_error_mood = None
#         print("✅ Mood vectorizer + model loaded")
#     except Exception as e:
#         load_error_mood = e
#         print(f"❌ Mood stack error: {e}")

# def load_amarzero_model():
#     global AMARZERO_CHECKPOINT, load_error_amarzero
#     try:
#         if AMARZERO_MODEL_PATH.exists():
#             AMARZERO_CHECKPOINT = torch.load(AMARZERO_MODEL_PATH, map_location=torch.device("cpu"))
#             print("✅ AmarZero checkpoint loaded")
#         else:
#             print(f"ℹ️ AmarZero checkpoint not found at {AMARZERO_MODEL_PATH}. Continuing without it.")
#         load_error_amarzero = None
#     except Exception as e:
#         load_error_amarzero = e
#         print(f"❌ AmarZero load error: {e}")

# @app.on_event("startup")
# def _startup():
#     load_symptom_stack()
#     load_brain_model()
#     load_heart_model()
#     load_mood_stack()
#     load_amarzero_model()
#     load_bone_model()

# # -----------------------------
# # Guards
# # -----------------------------
# def ensure_bone_ready():
#     if load_error_bone or BONE_MODEL is None:
#         raise HTTPException(status_code=500, detail=f"Bone fracture model not loaded: {load_error_bone}")

# def ensure_symptom_ready():
#     if load_error_symptom or SYMPTOM_MODEL is None or SYMPTOM_LABEL is None or not SYMPTOM_LIST:
#         raise HTTPException(status_code=500, detail=f"Symptom model not loaded: {load_error_symptom}")

# def ensure_brain_ready():
#     if load_error_brain or BRAIN_INTERPRETER is None:
#         raise HTTPException(status_code=500, detail=f"Brain model not loaded: {load_error_brain}")

# def ensure_heart_ready():
#     if load_error_heart or HEART_MODEL is None or HEART_SCALER is None:
#         raise HTTPException(status_code=500, detail=f"Heart model not loaded: {load_error_heart}")

# def ensure_mood_ready():
#     if load_error_mood or MOOD_VECTORIZER is None or MOOD_MODEL is None:
#         raise HTTPException(status_code=500, detail=f"Mood model not loaded: {load_error_mood}")

# def ensure_amarzero_ready():
#     if load_error_amarzero:
#         raise HTTPException(status_code=500, detail=f"AmarZero model load error: {load_error_amarzero}")

# # -----------------------------
# # Symptom Helpers
# # -----------------------------
# def _names_to_mask(user_symptoms: List[str]) -> List[bool]:
#     n = len(SYMPTOM_LIST)
#     mask = [False] * n
#     if not user_symptoms:
#         return mask

#     normalized_inputs = []
#     for s in user_symptoms:
#         if isinstance(s, str):
#             lang = detect_language(s)
#             if lang == "ar":
#                 t = try_translate_to_english(s) or s
#             else:
#                 t = s
#             normalized_inputs.append(_normalize_name(t))

#     expanded_targets: List[str] = []
#     for token in normalized_inputs:
#         expanded_targets.append(token)
#         if token in SYNONYMS:
#             expanded_targets.extend(SYNONYMS[token])

#     for t in expanded_targets:
#         tn = _normalize_name(t)
#         idx = NORM_TO_INDEX.get(tn)
#         if idx is not None:
#             mask[idx] = True
#     return mask

# def _predict(mask: List[bool]) -> Dict:
#     n_expected = int(getattr(SYMPTOM_MODEL, "n_features_in_", len(SYMPTOM_LIST)))
#     if len(mask) != n_expected:
#         raise HTTPException(status_code=500, detail=f"Internal mask length {len(mask)} != model expected {n_expected}")
#     features = np.array(mask, dtype=np.float32).reshape(1, -1)
#     pred_enc = SYMPTOM_MODEL.predict(features)
#     disease = SYMPTOM_LABEL.inverse_transform(pred_enc)[0]
#     proba = SYMPTOM_MODEL.predict_proba(features)[0]
#     order = np.argsort(proba)[::-1][:3]
#     classes = SYMPTOM_LABEL.classes_
#     top_three = [{"disease": str(classes[i]), "percent": f"{proba[i]*100:.2f}%"} for i in order]
#     return {"predicted_disease": str(disease), "top_three": top_three}

# # -----------------------------
# # AmarZero Helpers
# # -----------------------------
# def _extract_zip_bytes_to_tempdir(upload: UploadFile) -> str:
#     tmpdir = tempfile.mkdtemp(prefix="amarzero_")
#     zpath = os.path.join(tmpdir, "data.zip")
#     with open(zpath, "wb") as f:
#         f.write(upload.file.read())
#     with zipfile.ZipFile(zpath, "r") as zf:
#         zf.extractall(tmpdir)
#     return tmpdir

# def _find_patient_folder(root: str) -> Optional[str]:
#     entries = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
#     if not entries:
#         return None
#     return os.path.join(root, entries[0])

# def _check_brats_files(patient_path: str):
#     required = ["flair", "t1", "t1ce", "t2", "seg"]
#     present = []
#     for mod in required:
#         mod_files = [f for f in os.listdir(patient_path) if fnmatch.fnmatch(f, f"*{mod}.nii*")]
#         if mod_files:
#             present.append(mod)
#     return present

# def _load_segmentation(patient_path: str):
#     seg_files = [f for f in os.listdir(patient_path) if fnmatch.fnmatch(f, "*seg.nii*")]
#     if not seg_files:
#         raise FileNotFoundError("No *seg.nii* file found in patient folder.")
#     seg_path = os.path.join(patient_path, seg_files[0])
#     return nib.load(seg_path).get_fdata()

# def _extract_tumor_positions(seg_data: np.ndarray):
#     tumor_positions: List[List[float]] = []
#     labeled_array, num_features = ndimage.label(seg_data > 0)
#     if num_features <= 0:
#         return tumor_positions
#     sizes = ndimage.sum(seg_data > 0, labeled_array, range(1, num_features + 1))
#     main_label = int(np.argmax(sizes)) + 1
#     for z in range(seg_data.shape[2]):
#         slice_seg = labeled_array[:, :, z]
#         if np.any(slice_seg == main_label):
#             pos = np.argwhere(slice_seg == main_label)
#             centroid = np.mean(pos, axis=0)
#             tumor_positions.append([float(centroid[1]), float(centroid[0]), float(z)])
#     return tumor_positions

# def _tumor_stats_text(positions: List[List[float]]) -> str:
#     if not positions:
#         return "No tumor detected in segmentation."
#     arr = np.array(positions)
#     return (
#         f"- X: {np.min(arr[:,0]):.1f} → {np.max(arr[:,0]):.1f}\n"
#         f"- Y: {np.min(arr[:,1]):.1f} → {np.max(arr[:,1]):.1f}\n"
#         f"- Z: {np.min(arr[:,2]):.1f} → {np.max(arr[:,2]):.1f}\n"
#         f"- Mean (X,Y,Z): {np.mean(arr, axis=0)}"
#     )

# # -----------------------------
# # Routes: health / info
# # -----------------------------
# @app.get("/health")
# def health():
#     return {
#         "symptom_stack": "ok" if not load_error_symptom else str(load_error_symptom),
#         "brain_model": "ok" if not load_error_brain else str(load_error_brain),
#         "heart_stack": "ok" if not load_error_heart else str(load_error_heart),
#         "mood_stack": "ok" if not load_error_mood else str(load_error_mood),
#         "amarzero": "ok" if not load_error_amarzero else str(load_error_amarzero),
#         "model_dir": str(MODEL_DIR),
#         "symptom_count": len(SYMPTOM_LIST),
#         "model_features": getattr(SYMPTOM_MODEL, "n_features_in_", None),
#     }

# # -----------------------------
# # Routes: disease via symptoms
# # -----------------------------
# @app.post("/predict", response_model=PredictResponse)
# def predict(req: SymptomRequest = Body(...)):
#     ensure_symptom_ready()
#     mask = _names_to_mask(req.symptoms)
#     result = _predict(mask)
#     return PredictResponse(
#         predicted_disease=result["predicted_disease"],
#         top_three=[TopProbItem(**it) for it in result["top_three"]],
#     )

# # -----------------------------
# # Brain tumor (TFLite) & heart disease
# # -----------------------------
# @app.post("/predict/brain-tumor")
# def predict_brain_tumor(file: UploadFile = File(...)):
#     ensure_brain_ready()
#     contents = file.file.read()

#     # Preprocess with Pillow/NumPy
#     image = Image.open(BytesIO(contents)).convert("RGB").resize((224, 224))
#     arr_f32 = (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]  # (1,224,224,3)

#     # Quantize input if model expects quantized dtype
#     if BRAIN_IN_DTYPE in (np.uint8, np.int8) and BRAIN_IN_SCALE and BRAIN_IN_ZP is not None:
#         arr_for_input = np.round(arr_f32 / BRAIN_IN_SCALE + BRAIN_IN_ZP).astype(BRAIN_IN_DTYPE)
#     else:
#         arr_for_input = arr_f32.astype(np.float32)

#     # Inference
#     BRAIN_INTERPRETER.set_tensor(BRAIN_INPUT_INDEX, arr_for_input)
#     BRAIN_INTERPRETER.invoke()
#     outputs = BRAIN_INTERPRETER.get_tensor(BRAIN_OUTPUT_INDEX)  # (1,C) or (C,)

#     # Dequantize output if needed
#     if outputs.ndim == 2:
#         outputs = outputs[0]
#     if BRAIN_OUT_DTYPE in (np.uint8, np.int8) and BRAIN_OUT_SCALE and BRAIN_OUT_ZP is not None:
#         probs = (outputs.astype(np.float32) - BRAIN_OUT_ZP) * BRAIN_OUT_SCALE
#     else:
#         probs = outputs.astype(np.float32)

#     # If model outputs logits, apply softmax (safe fallback)
#     s = float(probs.sum())
#     if not (0.99 <= s <= 1.01) or (probs.min() < 0) or (probs.max() > 1):
#         exp = np.exp(probs - np.max(probs))
#         probs = exp / exp.sum()

#     idx = int(np.argmax(probs))
#     return {
#         "predicted_class": class_labels[idx],
#         "probability": float(probs[idx]),
#         "all_probabilities": {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
#     }
# @app.post("/predict/heart-disease", response_model=HeartDiseaseResponse)
# def predict_heart_disease(input_data: HeartDiseaseInput = Body(...)):
#     ensure_heart_ready()
#     feats = np.array([[input_data.gender_male, input_data.age_years,
#                        input_data.smoker_current, input_data.on_bp_meds,
#                        input_data.has_diabetes, input_data.cholesterol_total,
#                        input_data.bp_systolic, input_data.heart_rate_bpm]], dtype=np.float32)
#     scaled = HEART_SCALER.transform(feats)
#     pred_raw = HEART_MODEL.predict(scaled)[0]
#     proba_raw = HEART_MODEL.predict_proba(scaled)[0][1]
#     prediction = int(pred_raw)
#     probability = float(proba_raw)
#     result = "High 10-year risk" if prediction == 1 else "Low risk"
#     return HeartDiseaseResponse(result=result, probability=probability, prediction=prediction)
# # -----------------------------
# # Mood analysis endpoint
# # -----------------------------
# @app.post("/predict/mood", response_model=MoodResponse)
# def predict_mood(req: MoodRequest):
#     ensure_mood_ready()
#     raw = req.text or ""
#     lang = detect_language(raw)
#     translated_text: Optional[str] = None
#     text_for_model = raw

#     if lang != "en":
#         translated_text = try_translate_to_english(raw)
#         text_for_model = translated_text or raw

#     final_lang_for_clean = "en" if (lang == "en" or translated_text) else lang
#     cleaned = clean_text(text_for_model, lang=final_lang_for_clean)

#     if not cleaned.strip():
#         return MoodResponse(
#             mood="Unknown",
#             language=lang,
#             cleaned_text="",
#             translated_text=translated_text,
#             error="Empty text after cleaning; translation may have failed."
#         )

#     try:
#         vec = MOOD_VECTORIZER.transform([cleaned])
#         pred = MOOD_MODEL.predict(vec)[0]
#         mood = "Positive" if int(pred) == 4 else "Negative"
#         return MoodResponse(
#             mood=mood,
#             language=lang,
#             cleaned_text=cleaned,
#             translated_text=translated_text
#         )
#     except Exception as e:
#         return MoodResponse(
#             mood="Unknown",
#             language=lang,
#             cleaned_text=cleaned,
#             translated_text=translated_text,
#             error=f"Prediction error: {str(e)}"
#         )

# # -----------------------------
# # AmarZero endpoint
# # -----------------------------
# @app.post("/predict/amarzero", response_model=AmarZeroResponse)
# def predict_amarzero(zipfile_upload: UploadFile = File(...)):
#     ensure_amarzero_ready()
#     try:
#         root = _extract_zip_bytes_to_tempdir(zipfile_upload)
#         patient_folder = _find_patient_folder(root)
#         if not patient_folder:
#             return AmarZeroResponse(
#                 patient_folder=None,
#                 files_present=[],
#                 tumor_positions_count=0,
#                 stats=None,
#                 error="No patient folder found in ZIP."
#             )

#         present = _check_brats_files(patient_folder)
#         if "seg" not in present:
#             return AmarZeroResponse(
#                 patient_folder=os.path.basename(patient_folder),
#                 files_present=present,
#                 tumor_positions_count=0,
#                 stats=None,
#                 error="Segmentation file (*seg.nii*) not found."
#             )

#         seg = _load_segmentation(patient_folder)
#         positions = _extract_tumor_positions(seg)
#         stats = _tumor_stats_text(positions)
#         return AmarZeroResponse(
#             patient_folder=os.path.basename(patient_folder),
#             files_present=present,
#             tumor_positions_count=len(positions),
#             stats=stats,
#             error=None
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing ZIP: {str(e)}")

# # -----------------------------
# # Bone fracture endpoint
# # -----------------------------
# @app.post("/predict/bone-fracture", response_model=BoneFractureResponse)
# def predict_bone_fracture(file: UploadFile = File(...)):
#     ensure_bone_ready()
#     try:
#         image = Image.open(BytesIO(file.file.read())).convert("RGB")
#         tensor = BONE_TRANSFORM(image).unsqueeze(0)
#         with torch.no_grad():
#             outputs = BONE_MODEL(tensor)
#             probs = torch.softmax(outputs, dim=1).numpy()[0]
#         result = {BONE_CLASSES[i]: float(probs[i]) for i in range(len(BONE_CLASSES))}
#         predicted_class = BONE_CLASSES[int(np.argmax(probs))]
#         return BoneFractureResponse(predicted_class=predicted_class, probabilities=result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during bone fracture prediction: {str(e)}")

# # -----------------------------
# # Main execution
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=6000)
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import nltk


# from models.state import init_state
# from services.loaders import (
# load_symptom_stack,
# load_brain_model,
# load_heart_model,
# load_mood_stack,
# load_amarzero_model,
# load_bone_model,
# )
# from routers import health, symptoms, brain, heart, mood, amarzero, bone


# app = FastAPI(title="Medical Prediction Service", version="7.7.0")


# app.add_middleware(
# CORSMiddleware,
# allow_origins=["*"],
# allow_credentials=True,
# allow_methods=["*"],
# allow_headers=["*"],
# )



# for res, path in [("tokenizers/punkt", "punkt"), ("corpora/stopwords", "stopwords")]:
#     try:
#         nltk.data.find(res)
#     except LookupError:
#         nltk.download(path, quiet=True)



# init_state()


# # Routers
# app.include_router(health.router)
# app.include_router(symptoms.router)
# app.include_router(brain.router)
# app.include_router(heart.router)
# app.include_router(mood.router)
# app.include_router(amarzero.router)
# app.include_router(bone.router)




# @app.on_event("startup")
# def _startup():
#     load_symptom_stack()
#     load_brain_model()
#     load_heart_model()
#     load_mood_stack()
#     load_amarzero_model()
#     load_bone_model()
    
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------
# Load Models
# -----------------------------
MODEL_DIR = Path("./NasaModels")

reg_model = joblib.load(MODEL_DIR / "weather_reg_model.pkl")
class_model = joblib.load(MODEL_DIR / "weather_class_model.pkl")
unique_weathers = joblib.load(MODEL_DIR / "unique_weathers.pkl")
le_city = joblib.load(MODEL_DIR / "city_encoder.pkl")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Weather Prediction API", version="1.2.0")

# -----------------------------
# Schemas
# -----------------------------
class WeatherRequest(BaseModel):
    city: str
    datetime: str
    lat: float
    lon: float

class WeatherResponse(BaseModel):
    date: str              # NEW FIELD
    temperature: float
    humidity: float
    wind_speed: float
    weather: str

class WeeklyForecastRequest(BaseModel):
    city: str
    start_date: str
    lat: float
    lon: float

class WeeklyForecastResponse(BaseModel):
    city: str
    predictions: list[WeatherResponse]

# -----------------------------
# Helpers
# -----------------------------
def make_features(city: str, dt: datetime, lat: float, lon: float) -> pd.DataFrame:
    city_encoded = le_city.transform([city])[0]
    return pd.DataFrame({
        'city': [city_encoded],
        'lat': [lat],
        'lon': [lon],
        'hour': [dt.hour],
        'day': [dt.day],
        'month': [dt.month],
        'year': [dt.year],
        'dayofweek': [dt.weekday()]
    })

def predict_once(city: str, dt: datetime, lat: float, lon: float) -> WeatherResponse:
    data = make_features(city, dt, lat, lon)

    # Regression prediction
    reg_pred = reg_model.predict(data)[0]
    temp, humidity, wind_speed = float(reg_pred[0]), float(reg_pred[1]), float(reg_pred[2])

    # Classification prediction
    class_pred = class_model.predict(data)[0]
    weather = unique_weathers[int(class_pred)]

    return WeatherResponse(
        date=dt.strftime("%Y-%m-%d"),   # inject date
        temperature=temp,
        humidity=humidity,
        wind_speed=wind_speed,
        weather=weather
    )

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict", response_model=WeatherResponse)
def predict_weather(req: WeatherRequest):
    try:
        dt = pd.to_datetime(req.datetime)
        if req.city not in le_city.classes_:
            raise HTTPException(status_code=400, detail="Unsupported city.")

        return predict_once(req.city, dt, req.lat, req.lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_week", response_model=WeeklyForecastResponse)
def predict_week(req: WeeklyForecastRequest):
    try:
        start_dt = pd.to_datetime(req.start_date)
        if req.city not in le_city.classes_:
            raise HTTPException(status_code=400, detail="Unsupported city.")

        predictions = []
        for i in range(7):
            day_dt = start_dt + timedelta(days=i)
            forecast = predict_once(req.city, day_dt, req.lat, req.lon)
            predictions.append(forecast)

        return WeeklyForecastResponse(city=req.city, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2000)
