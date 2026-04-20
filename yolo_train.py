from ultralytics import YOLO
import os

def train_yolo():
    print("=== YOLO Training Started ===")


    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,  # GPU
        name="polyp_yolo"
    )

    print("=== Training Finished ===")
    best_model_path = "runs/detect/polyp_yolo/weights/best.pt"

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("best.pt not found!")

    print("=== Loading best.pt model ===")
    best_model = YOLO(best_model_path)

    print("=== Running Automatic Test Evaluation ===")
    test_results = best_model.val(
        data="dataset.yaml",
        split="test",
        imgsz=640,
        device=0
    )

    print("=== Test Evaluation Completed ===")
    print(test_results)

    print("\n The test results are saved in the blow folder:")
    print("runs/detect/polyp_yolo/")

if __name__ == "__main__":
    train_yolo()
