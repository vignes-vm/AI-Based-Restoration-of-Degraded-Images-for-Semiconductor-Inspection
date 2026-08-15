"""Export a trained NAFNet-SR checkpoint to TorchScript and ONNX for
deployment. TensorRT engine build is noted but not executed here (see
comment at bottom) since TensorRT is not installed in this environment.

Usage:
    python -m src.eval.export_model --checkpoint weights/nafnet_sr_best.pth \
        --out_dir weights/exported
"""
import argparse
import os

import torch
import yaml

from src.models.nafnet_sr import NAFNetSR


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    m, d = cfg["model"], cfg["data"]
    model = NAFNetSR(in_channels=d["in_channels"], width=m["width"], enc_blocks=m["enc_blocks"],
                      middle_blocks=m["middle_blocks"], dec_blocks=m["dec_blocks"],
                      scale_factor=d["scale_factor"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def export_torchscript(model, dummy_input, out_path):
    traced = torch.jit.trace(model, dummy_input)
    traced.save(out_path)
    print(f"[export] TorchScript saved -> {out_path}")


def export_onnx(model, dummy_input, out_path):
    torch.onnx.export(
        model, dummy_input, out_path,
        input_names=["noisy_lr"], output_names=["restored"],
        dynamic_axes={"noisy_lr": {0: "batch", 2: "height", 3: "width"},
                       "restored": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
    )
    print(f"[export] ONNX saved -> {out_path}")


def try_export_tensorrt(onnx_path, out_path):
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        print("[export] TensorRT not installed — skipping engine build. "
              "To build a TensorRT engine on a machine with TensorRT available, run:\n"
              f"    trtexec --onnx={onnx_path} --saveEngine={out_path} --fp16")
        return
    print("[export] TensorRT is available but automated engine build is not wired up here; "
          f"run `trtexec --onnx={onnx_path} --saveEngine={out_path} --fp16` manually.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/nafnet_sr_best.pth")
    parser.add_argument("--out_dir", default="weights/exported")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, cfg = load_model(args.checkpoint)

    d = cfg["data"]
    lr_size = d["gt_size"] // d["scale_factor"]
    dummy_input = torch.randn(1, d["in_channels"], lr_size, lr_size)

    ts_path = os.path.join(args.out_dir, "nafnet_sr.torchscript.pt")
    onnx_path = os.path.join(args.out_dir, "nafnet_sr.onnx")
    trt_path = os.path.join(args.out_dir, "nafnet_sr.trt")

    export_torchscript(model, dummy_input, ts_path)
    export_onnx(model, dummy_input, onnx_path)
    try_export_tensorrt(onnx_path, trt_path)


if __name__ == "__main__":
    main()
