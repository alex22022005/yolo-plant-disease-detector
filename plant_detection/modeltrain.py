import os
import subprocess
from roboflow import Roboflow

def main():
    """
    Main function to handle the entire training pipeline:
    1. Download dataset from Roboflow.
    2. Run YOLOv8 training using the local GPU.
    3. Print the location of the final trained model.
    """
    print("🚀 Step 1: Downloading dataset from Roboflow...")
    
    # --- IMPORTANT: PASTE YOUR ROBOFLOW API KEY HERE ---
    ROBOFLOW_API_KEY = "GUeccORHJBVBaiFUIxSy"

    try:
        # Authenticate with Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # Get your project and version
        project = rf.workspace("smart-drone").project("plant-disease-detection-ynk3q")
        version = project.version(15)
        
        # Download the dataset to a local folder
        dataset = version.download("yolov11")
        
        print(f"✅ Dataset downloaded successfully to: {dataset.location}")
        print("-" * 60)

    except Exception as e:
        print(f"❌ An error occurred during dataset download: {e}")
        return # Exit if download fails

    # --- Training Configuration ---
    project_name = 'plant_disease_training_local'
    run_name = 'first_local_run'
    epochs = 75
    
    print("🧠 Step 2: Starting model training using your local GPU...")

    # Construct the training command
    command = [
        "yolo",
        "task=detect",
        "mode=train",
        "model=yolov8n.pt",
        f"data={os.path.join(dataset.location, 'data.yaml')}",
        f"epochs={epochs}",
        "imgsz=640",
        f"project={project_name}",
        f"name={run_name}"
    ]

    try:
        # Run the command using subprocess
        subprocess.run(command, check=True)
        print("🎉 Training complete!")
        print("-" * 60)

    except subprocess.CalledProcessError as e:
        print(f"❌ An error occurred during training: {e}")
        return
    except FileNotFoundError:
        print("❌ Error: 'yolo' command not found.")
        print("   Please ensure 'ultralytics' is installed correctly.")
        return

    # --- Locate the Final Model ---
    print("📦 Step 3: Locating the trained model...")
    
    output_path = os.path.join(os.getcwd(), project_name, run_name, 'weights', 'best.pt')

    if os.path.exists(output_path):
        print("\n" + "="*60)
        print("✅ Success! Your trained model is ready.")
        print(f"   You can find it at: {os.path.abspath(output_path)}")
        print("="*60)
    else:
        print(f"❌ Could not find the trained model at the expected path: {output_path}")


if __name__ == "__main__":
    main()