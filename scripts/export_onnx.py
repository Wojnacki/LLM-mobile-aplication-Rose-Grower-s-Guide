from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_NAME   = "intfloat/multilingual-e5-small"
OUTPUT_DIR   = Path("models/onnx/e5-small-fp32")
QUANT_DIR    = Path("models/onnx/e5-small-int8")

def export():
    print("Eksport do ONNX (fp32)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Zapisano fp32: {OUTPUT_DIR}")

def quantize():
    print("Kwantyzacja do int8...")
    QUANT_DIR.mkdir(parents=True, exist_ok=True)

    input_path  = OUTPUT_DIR / "model.onnx"
    output_path = QUANT_DIR  / "model.onnx"

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    # Skopiuj tokenizer do katalogu int8
    import shutil
    for f in OUTPUT_DIR.iterdir():
        if f.suffix != ".onnx":
            shutil.copy(f, QUANT_DIR / f.name)

    size_fp32 = input_path.stat().st_size  / 1024**2
    size_int8 = output_path.stat().st_size / 1024**2
    print(f"fp32: {size_fp32:.1f} MB  →  int8: {size_int8:.1f} MB")
    print(f"Redukcja: {(1 - size_int8/size_fp32)*100:.0f}%")

def verify():
    print("\nWeryfikacja modelu int8...")
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer
    import logging
    # Wycisz fałszywy warning o regex tokenizera
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(str(QUANT_DIR))
    sess = ort.InferenceSession(
        str(QUANT_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )

    # Sprawdź jakich inputów faktycznie wymaga model
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"Wymagane inputy modelu: {input_names}")

    test = tokenizer(
        "query: jak podlewać róże",
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Podaj tylko te inputy których model faktycznie wymaga
    import numpy as np

    feed = {k: v for k, v in test.items() if k in input_names}

    # Jeśli model wymaga token_type_ids a tokenizer go nie zwrócił — dodaj zera
    if "token_type_ids" in input_names and "token_type_ids" not in feed:
        feed["token_type_ids"] = np.zeros_like(test["input_ids"])

    outputs = sess.run(None, feed)

    # Mean pooling
    token_embeddings = outputs[0]
    attention_mask   = test["attention_mask"]
    mask_expanded    = attention_mask[..., np.newaxis].astype(np.float32)
    pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clip(min=1e-9)

    # Normalizacja
    norm   = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / norm

    print(f"Wymiar embeddingu: {pooled.shape[1]}")
    print(f"Norma wektora:     {np.linalg.norm(pooled[0]):.4f}  (powinno być ~1.0)")
    print("Weryfikacja OK ✅")

if __name__ == "__main__":
    export()
    quantize()
    verify()