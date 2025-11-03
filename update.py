from roboflow import Roboflow

rf = Roboflow(api_key="oan9JE4IqJkKgSqR5YCc")
workspace = rf.workspace("pig")

workspace.deploy_model(
    model_type="yolov12",
    model_path="runs/train/exp11",
    project_ids=[
        "2",
    ],
    model_name="my-pig-model",
)
