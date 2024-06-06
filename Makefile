point2plane:
	python scripts/optimize.py \
	model=point2plane \
	tags=["point2plane"] \
	model.lr=1e-02 \
	data=kinect \
	data.start_frame_idx=19 \
	model.optimize_frames=10 \
	data.batch_size=10 \
	logger=wandb \
	model.init_mode=kinect \
	model.vertices_mask=full \
	model.save_interval=10 \
	trainer.max_epochs=801 \
	callbacks.finetune_scheduler.milestones=[0,100,150,450] \
	callbacks.finetune_scheduler.params=["global_pose|transl","neck_pose|eye_pose","shape_params","expression_params"] \
	callbacks.coarse2fine_scheduler.milestones=[0,100,101,150,151,450,451,800] \
	callbacks.coarse2fine_scheduler.scales=[0.125,1.0,0.125,1.0,0.125,1.0,0.125,1.0] \


joint_point2plane:
	python scripts/optimize.py \
	tags=["point2plane"] \
	logger=wandb \
	model=point2plane \
	model.lr=1e-02 \
	model.init_mode=kinect \
	model.vertices_mask=full \
	model.optimize_frames=6 \
	model.save_interval=10 \
	data=kinect \
	data.batch_size=5  \
	data.start_frame_idx=19 \
	trainer.max_epochs=801 \
	trainer.accelerator=gpu \
	callbacks.coarse2fine_scheduler.milestones=[0,800] \
	callbacks.coarse2fine_scheduler.scales=[0.125,1.0] \
	callbacks.finetune_scheduler.milestones=[0,100,150,450] \
	callbacks.finetune_scheduler.params=["global_pose|transl","neck_pose|eye_pose","shape_params","expression_params"] \

point2plane_flame:
	python scripts/optimize.py \
	model=point2plane \
	data=flame \
	logger=wandb \
	model.lr=1e-02 \
	model.init_mode=flame \
	model.vertices_mask=full \
	trainer.max_epochs=200 \
	trainer.accelerator=gpu \
	callbacks.coarse2fine_scheduler.milestones=[0,150] \
	callbacks.coarse2fine_scheduler.scales=[0.1,0.25] \


create_video:
	python scripts/create_video.py +framerate=30 +video_dir="logs/optimize/runs/2024-06-06_10-29-05/render_normal" +video_path="temp/render_normal.mp4"
	python scripts/create_video.py +framerate=30 +video_dir="logs/optimize/runs/2024-06-06_10-29-05/batch_color" +video_path="temp/batch_color.mp4"