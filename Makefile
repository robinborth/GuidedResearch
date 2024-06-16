point2plane:
	python scripts/optimize.py \
	logger=wandb \
	tags=["point2plane"] \
	model=flame \
	model.init_mode=kinect \
	model.vertices_mask=full \
	model.optimize_frames=20 \
	data=kinect \
	data.batch_size=20  \
	data.start_frame_idx=19 \
	trainer.max_iters=20 \
	trainer.max_optims=100 \
	trainer.save_interval=1 \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.finetune.milestones=[0,3,5] \
	scheduler.finetune.params=[["global_pose","transl"],["neck_pose","eye_pose"],["shape_params","expression_params"]] \
	scheduler.finetune.lr=[[1e-02,1e-02],[1e-02,1e-02],[1e-02,1e-02]]

profile:
	python scripts/optimize.py \
	tags=["point2plane"] \
	model=flame \
	model.lr=1e-02 \
	model.init_mode=kinect \
	model.vertices_mask=full \
	model.optimize_frames=20 \
	model.save_interval=1 \
	data=kinect \
	data.batch_size=20  \
	data.start_frame_idx=19 \
	trainer.max_epochs=25 \
	trainer.accelerator=gpu \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[1] \
	scheduler.finetune.milestones=[0] \
	scheduler.finetune.params=["global_pose|transl|neck_pose|eye_pose|shape_params|expression_params"] \

create_video:
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_normal" +video_path="temp/render_normal.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/batch_color" +video_path="temp/batch_color.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/error_point_to_plane" +video_path="temp/error_point_to_plane.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_merged" +video_path="temp/render_merged.mp4"