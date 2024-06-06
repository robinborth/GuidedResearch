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
	model.optimize_frames=4 \
	model.save_interval=10 \
	data=kinect \
	data.batch_size=4 \
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