"""
Gradio UI for the Clinical 3D Pipeline.
Calls the FastAPI backend (configurable base URL). Start the API first, then launch this UI.
"""

import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx
import numpy as np
from PIL import Image as PILImage

API_URL = os.getenv("PIPELINE_API_URL", "http://127.0.0.1:8000")


def _get(url: str, **kwargs: Any) -> httpx.Response:
    with httpx.Client(timeout=60.0) as client:
        return client.get(url, **kwargs)


def _post(url: str, **kwargs: Any) -> httpx.Response:
    with httpx.Client(timeout=120.0) as client:
        return client.post(url, **kwargs)


# --- Step 1: Create case ---
def create_case(
    scan_file: Optional[gr.File],
    report_text: str,
    modality: str,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    if not scan_file or not report_text.strip():
        return "Upload a scan and enter report text.", state, ""
    path = scan_file if isinstance(scan_file, str) else (scan_file.name if hasattr(scan_file, "name") else None)
    if not path or not os.path.isfile(path):
        return "Please select a valid scan file.", state, ""
    try:
        with open(path, "rb") as f:
            content = f.read()
        filename = os.path.basename(path)
        r = _post(
            f"{API_URL}/cases",
            files={"scan_file": (filename, content)},
            data={"modality": modality, "report_text": report_text.strip()},
        )
        r.raise_for_status()
        data = r.json()
        case_id = data["case_id"]
        scan = data.get("scan", {})
        intent = data.get("intent", {})
        shape = scan.get("shape", [0, 0, 0])
        state = {**state, "case_id": case_id, "volume_shape": shape}
        summary = f"**Case ID:** `{case_id}`\n\n**Shape (z,y,x):** {shape}\n\n**Intent:**\n```json\n{json.dumps(intent, indent=2)}\n```"
        return summary, state, case_id
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} – {e.response.text}", state, state.get("case_id", "")
    except Exception as e:
        return f"Error: {e}", state, state.get("case_id", "")


# --- Step 2: Ground ---
def run_ground(case_id: str, organ: Optional[str], state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not case_id.strip():
        return "Enter or select a case ID.", state
    try:
        params = {"organ": organ} if organ else {}
        r = _post(f"{API_URL}/cases/{case_id.strip()}/ground", params=params)
        r.raise_for_status()
        bbox = r.json()
        state = {**state, "case_id": case_id.strip()}
        return f"**Bounding box:**\n```json\n{json.dumps(bbox, indent=2)}\n```", state
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} – {e.response.text}", state
    except Exception as e:
        return f"Error: {e}", state


# --- Step 3: Human verification ---
def load_slice(
    case_id: str,
    axis: str,
    index: int,
    with_bbox: bool,
    radiological_display: bool,
    state: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if not case_id.strip():
        return None, state
    try:
        r = _get(
            f"{API_URL}/cases/{case_id.strip()}/volume_slice",
            params={
                "axis": axis,
                "index": index,
                "with_bbox": with_bbox,
                "radiological_display": radiological_display,
            },
        )
        r.raise_for_status()
        arr = np.array(PILImage.open(io.BytesIO(r.content)).convert("L"))
        h, w = arr.shape[0], arr.shape[1]
        vol_shape = state.get("volume_shape") or get_volume_info(case_id)
        state = {
            **state,
            "slice_img_shape": (w, h),
            "case_id": case_id.strip(),
            "volume_shape": vol_shape,
            "radiological_display": radiological_display,
            "_slice_error": "",
        }
        return arr, state, ""
    except httpx.HTTPStatusError as e:
        err = f"⚠️ Slice load error {e.response.status_code}: {e.response.text[:300]}"
        return None, {**state, "_slice_error": err}, err
    except Exception as e:
        err = f"⚠️ Slice load error: {e}"
        return None, {**state, "_slice_error": err}, err


def on_slice_image_click(evt: gr.SelectData, state: Dict[str, Any]) -> Dict[str, Any]:
    """Store last click in state for Confirm button."""
    px, py = evt.index[0], evt.index[1]
    return {**state, "last_click_px": px, "last_click_py": py}


def confirm_point_from_state(
    case_id: str,
    axis: str,
    slice_index: int,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    if not case_id.strip():
        return "Enter a case ID.", state
    px = state.get("last_click_px")
    py = state.get("last_click_py")
    if px is None or py is None:
        return "Click on the slice image to set the point, then click 'Confirm this point'.", state
    img_shape = state.get("slice_img_shape") or (512, 512)
    img_w, img_h = img_shape[0], img_shape[1]
    radiological = state.get("radiological_display", False)
    try:
        r = _post(
            f"{API_URL}/cases/{case_id.strip()}/pixel-to-voxel",
            json={
                "px": float(px),
                "py": float(py),
                "axis": axis,
                "index": int(slice_index),
                "img_width": img_w,
                "img_height": img_h,
                "radiological_display": radiological,
            },
        )
        r.raise_for_status()
        pt = r.json()
        x, y, z = int(pt["x"]), int(pt["y"]), int(pt["z"])
        r2 = _post(
            f"{API_URL}/cases/{case_id.strip()}/confirm-point",
            json={"x": x, "y": y, "z": z},
        )
        r2.raise_for_status()
        ver = _get(f"{API_URL}/cases/{case_id.strip()}/verification").json()
        state = {**state, "case_id": case_id.strip(), "last_confirm": (x, y, z)}
        return f"**Confirmed point (voxel):** ({x}, {y}, {z})\n\n**Verification:**\n```json\n{json.dumps(ver, indent=2)}\n```", state
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} – {e.response.text}", state
    except Exception as e:
        return f"Error: {e}", state


# --- Step 4: Segment ---
def run_segment(case_id: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not case_id.strip():
        return "Enter or select a case ID.", state
    try:
        r = _post(f"{API_URL}/cases/{case_id.strip()}/segment")
        r.raise_for_status()
        data = r.json()
        state = {**state, "case_id": case_id.strip()}
        return f"**Result:**\n```json\n{json.dumps(data, indent=2)}\n```", state
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} – {e.response.text}", state
    except Exception as e:
        return f"Error: {e}", state


# --- Step 5: Analyze ---
def run_analyze(case_id: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not case_id.strip():
        return "Enter or select a case ID.", state
    try:
        r = _post(f"{API_URL}/cases/{case_id.strip()}/analyze")
        r.raise_for_status()
        data = r.json()
        state = {**state, "case_id": case_id.strip()}
        return f"**Result:**\n```json\n{json.dumps(data, indent=2)}\n```", state
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} – {e.response.text}", state
    except Exception as e:
        return f"Error: {e}", state


def get_volume_info(case_id: str) -> List[int]:
    if not case_id.strip():
        return [1, 1, 1]
    try:
        r = _get(f"{API_URL}/cases/{case_id.strip()}/volume_info")
        r.raise_for_status()
        return r.json().get("shape", [1, 1, 1])
    except Exception:
        return [1, 1, 1]


def slice_slider_max(case_id: str, axis: str) -> gr.Slider:
    shape = get_volume_info(case_id)
    if axis == "z" and len(shape) > 0:
        mx = max(0, shape[0] - 1)
    elif axis == "y" and len(shape) > 1:
        mx = max(0, shape[1] - 1)
    elif axis == "x" and len(shape) > 2:
        mx = max(0, shape[2] - 1)
    else:
        mx = 512
    return gr.update(maximum=mx, value=0)


def build_ui() -> gr.Blocks:
    state = gr.State({})
    with gr.Blocks(title="Clinical 3D Pipeline") as app:
        gr.Markdown("# Clinical 3D Pipeline")
        gr.Markdown("Start the FastAPI backend first (`uvicorn main:app --reload --port 8000`), then use this UI.")
        case_id_shared = gr.Textbox(label="Case ID", placeholder="Set by Create case or enter manually", interactive=True)

        with gr.Tab("1. Create case"):
            scan_file = gr.File(label="Scan (ZIP / NIfTI / NRRD / image)", type="filepath")
            report_text = gr.Textbox(label="Report text", placeholder="Paste radiology report...", lines=4)
            modality = gr.Dropdown(choices=["ct", "mri", "xray", "us"], value="ct", label="Modality")
            create_btn = gr.Button("Create case")
            create_out = gr.Markdown()
            create_btn.click(
                create_case,
                inputs=[scan_file, report_text, modality, state],
                outputs=[create_out, state, case_id_shared],
            )

        with gr.Tab("2. Ground"):
            organ_drop = gr.Dropdown(label="Organ (optional)", choices=["kidney_right", "kidney_left", "liver", "spleen"], value="kidney_right")
            ground_btn = gr.Button("Run grounding")
            ground_out = gr.Markdown()
            ground_btn.click(
                run_ground,
                inputs=[case_id_shared, organ_drop, state],
                outputs=[ground_out, state],
            )

        with gr.Tab("3. Human verification (slice + click)"):
            gr.Markdown(
                "**Doctor as validator (human-in-the-loop):** The AI gives a rough bounding box from Step 2. "
                "You see the CT slice with the box. If the box is off or the report points to a specific lesion, "
                "**click the exact center of the mass/nodule** on the image, then click **Confirm this point**. "
                "That confirmed (x, y, z) is the safety signal—segmentation runs only after you verify."
            )
            axis_radio = gr.Radio(choices=["z", "y", "x"], value="z", label="Axis")
            slice_slider = gr.Slider(0, 512, value=0, step=1, label="Slice index")
            with_bbox_check = gr.Checkbox(value=True, label="Show bbox on slice")
            radiological_check = gr.Checkbox(
                value=False,
                label="Radiological display (left on screen = patient left; use same setting when confirming)",
            )
            slice_img = gr.Image(label="Slice (click to set point)", type="numpy")
            verify_btn = gr.Button("Confirm this point")
            verify_out = gr.Markdown()
            # Update slider max when case_id or axis changes; when axis changes also reload slice
            def update_slider_and_slice(cid, ax, idx, wb, rad, st):
                slider_upd = slice_slider_max(cid, ax)
                img, new_st, err = load_slice(cid, ax, int(idx), wb, rad, st)
                return slider_upd, img, new_st, err
            case_id_shared.change(
                update_slider_and_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_slider, slice_img, state, gr.Markdown()],
            )
            axis_radio.change(
                update_slider_and_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_slider, slice_img, state, gr.Markdown()],
            )
            # Load slice: returns (image, state, error_msg) to capture slice_img_shape
            def do_load_slice(cid, ax, idx, wb, rad, st):
                return load_slice(cid, ax, int(idx), wb, rad, st)
            slice_load_btn = gr.Button("Load slice")
            slice_err_out = gr.Markdown(visible=True, value="")
            slice_load_btn.click(
                do_load_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_img, state, slice_err_out],
            )
            slice_slider.change(
                do_load_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_img, state, slice_err_out],
            )
            with_bbox_check.change(
                do_load_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_img, state, slice_err_out],
            )
            radiological_check.change(
                do_load_slice,
                inputs=[case_id_shared, axis_radio, slice_slider, with_bbox_check, radiological_check, state],
                outputs=[slice_img, state, slice_err_out],
            )
            slice_img.select(on_slice_image_click, inputs=[state], outputs=[state])
            verify_btn.click(
                confirm_point_from_state,
                inputs=[case_id_shared, axis_radio, slice_slider, state],
                outputs=[verify_out, state],
            )

        with gr.Tab("4. Segment"):
            seg_btn = gr.Button("Run segmentation")
            seg_out = gr.Markdown()
            seg_btn.click(run_segment, inputs=[case_id_shared, state], outputs=[seg_out, state])

        with gr.Tab("5. Analyze"):
            analyze_btn = gr.Button("Run mesh & metrics")
            analyze_out = gr.Markdown()
            analyze_btn.click(run_analyze, inputs=[case_id_shared, state], outputs=[analyze_out, state])

    return app


def main() -> None:
    app = build_ui()
    for port in [7860, 7861, 7862]:
        try:
            app.launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft())
            break
        except OSError:
            if port == 7862:
                raise


if __name__ == "__main__":
    main()
