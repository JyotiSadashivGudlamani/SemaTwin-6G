import os
import sys

if sys.platform == "win32":
    try:
        torch_lib_path = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.app import (
    get_data_module,
    get_resnet_model,
    get_feature_extractor,
    get_transmission_pipeline,
    update_shared_experiment_image
)
from digital_twin.twin import DigitalTwinEngine

def test_flow():
    print("=== TESTING RESULTS PAGE DATA FLOW ===")
    
    # Mock streamlit session state dictionary
    class SessionState(dict):
        def __getattr__(self, key):
            return self.get(key, None)
        def __setattr__(self, key, value):
            self[key] = value

    import streamlit as st
    st.session_state = SessionState()
    
    dm = get_data_module()
    model = get_resnet_model()
    extractor = get_feature_extractor()
    pipeline = get_transmission_pipeline()
    try:
        import torch
        torch.manual_seed(42)
    except Exception:
        pass
    dt_engine = DigitalTwinEngine()
    st.session_state["dt_engine"] = dt_engine

    # -------------------------------------------------------------
    # TEST 1: Image = automobile, SNR = 5 dB, Channel = AWGN
    # -------------------------------------------------------------
    print("\n--- TEST 1: Automobile | SNR = 5 dB | Channel = AWGN ---")
    pil_auto, cls_auto, _, _, path_auto = dm.get_correctly_classified_sample_by_class("automobile", model=model)
    update_shared_experiment_image(pil_auto, cls_auto, path_auto, model, extractor)
    
    st.session_state["selected_snr_db"] = 5
    st.session_state["comm_channel_type"] = "AWGN"
    st.session_state["selected_node"] = "N3"

    # Simulate Communication page transmission
    target_telemetry = dict(dt_engine.state.node_telemetry.get("N3", {}))
    target_telemetry["snr_db"] = 5.0
    res_1 = pipeline.transmit_real_jscc(
        pil_or_np_img=st.session_state["selected_image"],
        destination_node_id="N3",
        node_telemetry=target_telemetry,
        dt_engine=dt_engine,
        semantic_meaning="AUTOMOBILE",
        channel_type="AWGN"
    )
    img_class_id = "automobile"
    res_1["image_id"] = img_class_id
    st.session_state["last_jscc_result"] = res_1
    st.session_state["results_awgn"] = res_1
    st.session_state["tested_measurements"] = [{
        "image_id": img_class_id,
        "snr": 5,
        "channel": "AWGN",
        "mse": float(res_1["mse"]),
        "psnr": float(res_1["psnr_db"]),
        "node": "N3"
    }]

    print(f"TEST 1 Image: {st.session_state['actual_class']} (ID: {img_class_id})")
    print(f"TEST 1 SNR: {st.session_state['selected_snr_db']} dB")
    print(f"TEST 1 Channel: {st.session_state['comm_channel_type']}")
    print(f"TEST 1 Measured MSE: {res_1['mse']:.6f}")
    print(f"TEST 1 Measured PSNR: {res_1['psnr_db']:.2f} dB")
    assert st.session_state['actual_class'].lower() == "automobile"
    assert res_1['snr_db'] == 5.0
    assert res_1['channel_type'] == "AWGN"
    print("[OK] TEST 1 PASSED!")

    # -------------------------------------------------------------
    # TEST 2: Change SNR = 10 dB
    # -------------------------------------------------------------
    print("\n--- TEST 2: Automobile | SNR = 10 dB | Channel = AWGN ---")
    st.session_state["selected_snr_db"] = 10

    target_telemetry["snr_db"] = 10.0
    res_2 = pipeline.transmit_real_jscc(
        pil_or_np_img=st.session_state["selected_image"],
        destination_node_id="N3",
        node_telemetry=target_telemetry,
        dt_engine=dt_engine,
        semantic_meaning="AUTOMOBILE",
        channel_type="AWGN"
    )
    res_2["image_id"] = img_class_id
    st.session_state["last_jscc_result"] = res_2
    st.session_state["results_awgn"] = res_2
    st.session_state["tested_measurements"].append({
        "image_id": img_class_id,
        "snr": 10,
        "channel": "AWGN",
        "mse": float(res_2["mse"]),
        "psnr": float(res_2["psnr_db"]),
        "node": "N3"
    })

    print(f"TEST 2 Image: {st.session_state['actual_class']}")
    print(f"TEST 2 SNR: {st.session_state['selected_snr_db']} dB")
    print(f"TEST 2 Measured MSE: {res_2['mse']:.6f}")
    print(f"TEST 2 Measured PSNR: {res_2['psnr_db']:.2f} dB")
    auto_pts = [p for p in st.session_state["tested_measurements"] if p["image_id"] == "automobile"]
    print(f"Automobile Graph Points ({len(auto_pts)}): {auto_pts}")
    assert len(auto_pts) == 2
    assert isinstance(res_2['psnr_db'], (float, int))
    print("[OK] TEST 2 PASSED!")

    # -------------------------------------------------------------
    # TEST 3: Change Image = dog
    # -------------------------------------------------------------
    print("\n--- TEST 3: Dog | SNR = 5 dB | Channel = AWGN ---")
    pil_dog, cls_dog, _, _, path_dog = dm.get_correctly_classified_sample_by_class("dog", model=model)
    update_shared_experiment_image(pil_dog, cls_dog, path_dog, model, extractor)
    st.session_state["selected_snr_db"] = 5
    st.session_state["comm_channel_type"] = "AWGN"

    target_telemetry["snr_db"] = 5.0
    res_3 = pipeline.transmit_real_jscc(
        pil_or_np_img=st.session_state["selected_image"],
        destination_node_id="N3",
        node_telemetry=target_telemetry,
        dt_engine=dt_engine,
        semantic_meaning="DOG",
        channel_type="AWGN"
    )
    dog_class_id = "dog"
    res_3["image_id"] = dog_class_id
    st.session_state["last_jscc_result"] = res_3
    st.session_state["results_awgn"] = res_3
    st.session_state["tested_measurements"].append({
        "image_id": dog_class_id,
        "snr": 5,
        "channel": "AWGN",
        "mse": float(res_3["mse"]),
        "psnr": float(res_3["psnr_db"]),
        "node": "N3"
    })

    print(f"TEST 3 Image: {st.session_state['actual_class']}")
    dog_pts = [p for p in st.session_state["tested_measurements"] if p["image_id"] == "dog"]
    auto_pts_isolated = [p for p in st.session_state["tested_measurements"] if p["image_id"] == "automobile"]
    print(f"Dog Graph Points ({len(dog_pts)}): {dog_pts}")
    print(f"Automobile Graph Points (Isolated): {len(auto_pts_isolated)}")
    assert len(dog_pts) == 1
    assert len(auto_pts_isolated) == 2
    assert st.session_state['actual_class'].lower() == "dog"
    print("[OK] TEST 3 PASSED!")

    # -------------------------------------------------------------
    # TEST 4: Change Channel = Rayleigh Fading
    # -------------------------------------------------------------
    print("\n--- TEST 4: Dog | SNR = 5 dB | Channel = Rayleigh Fading ---")
    st.session_state["comm_channel_type"] = "Rayleigh"

    res_4 = pipeline.transmit_real_jscc(
        pil_or_np_img=st.session_state["selected_image"],
        destination_node_id="N3",
        node_telemetry=target_telemetry,
        dt_engine=dt_engine,
        semantic_meaning="DOG",
        channel_type="Rayleigh"
    )
    res_4["image_id"] = dog_class_id
    st.session_state["last_jscc_result"] = res_4
    st.session_state["results_rayleigh"] = res_4
    st.session_state["tested_measurements"].append({
        "image_id": dog_class_id,
        "snr": 5,
        "channel": "Rayleigh",
        "mse": float(res_4["mse"]),
        "psnr": float(res_4["psnr_db"]),
        "node": "N3"
    })

    print(f"TEST 4 Channel: {st.session_state['comm_channel_type']}")
    print(f"TEST 4 Measured Rayleigh MSE: {res_4['mse']:.6f}")
    print(f"TEST 4 Measured Rayleigh PSNR: {res_4['psnr_db']:.2f} dB")
    dog_ray_pts = [p for p in st.session_state["tested_measurements"] if p["image_id"] == "dog" and p["channel"] == "Rayleigh"]
    dog_awgn_pts = [p for p in st.session_state["tested_measurements"] if p["image_id"] == "dog" and p["channel"] == "AWGN"]
    assert len(dog_ray_pts) == 1
    assert len(dog_awgn_pts) == 1
    assert res_4['channel_type'] == "Rayleigh"
    print("[OK] TEST 4 PASSED!")

    print("\n[OK] ALL 4 TEST CASES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_flow()
