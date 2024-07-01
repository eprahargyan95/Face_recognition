from roboflow import Roboflow

rf = Roboflow(api_key="5BrGlZj9T3HsNh7NX6wy")
project = rf.workspace().project("tesis-1xzsh")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "YOUR_VIDEO.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)
