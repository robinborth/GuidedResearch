point2plane:
	python scripts/icp_optimize.py \
	logger=wandb \
	tags=["point2plane"] \
	model=point2plane \
	model.lr=1e-02 \
	model.init_mode=kinect \
	model.vertices_mask=full \
	model.optimize_frames=1 \
	model.save_interval=20 \
	data=kinect \
	data.batch_size=1  \
	data.start_frame_idx=19 \
	trainer.max_epochs=741 \
	trainer.accelerator=gpu \
	+trainer.accumulate_grad_batches=1 \
	scheduler.coarse2fine.milestones=[0,740] \
	scheduler.coarse2fine.scales=[0.125,1.0] \
	scheduler.finetune.milestones=[0,100,150] \
	scheduler.finetune.params=["global_pose|transl","neck_pose|eye_pose","shape_params|expression_params"] \

create_video:
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_normal" +video_path="temp/render_normal.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/batch_color" +video_path="temp/batch_color.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/error_point_to_plane" +video_path="temp/error_point_to_plane.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_merged" +video_path="temp/render_merged.mp4"