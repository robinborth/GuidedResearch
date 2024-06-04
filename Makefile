point2point:
	python scripts/optimize.py \
	model=point2point \
	data=kinect \
	logger=wandb \
	model.lr=1e-03 \
	model.init_mode=kinect \
	model.vertices_mask=face \
	trainer.max_epochs=150 \
	callbacks.coarse2fine_scheduler.milestones=[0,100,130,140] \
	callbacks.coarse2fine_scheduler.scales=[0.05,0.1,0.25,0.5] \

point2point_shape:
	python scripts/optimize.py \
	model=point2point \
	data=kinect \
	logger=wandb \
	model.lr=1e-03 \
	model.init_mode=kinect \
	model.vertices_mask=face \
	trainer.max_epochs=150 \
	callbacks.coarse2fine_scheduler.milestones=[0,120] \
	callbacks.coarse2fine_scheduler.scales=[0.1,0.25] \
	callbacks.finetune_scheduler.milestones=[0,50] \
	callbacks.finetune_scheduler.params=["global_pose|transl","shape_params"] \


point2plane:
	python scripts/optimize.py \
	model=point2plane \
	data=kinect \
	logger=wandb \
	model.lr=1e-02 \
	model.init_mode=kinect \
	model.vertices_mask=face \
	trainer.max_epochs=100 \
	callbacks.coarse2fine_scheduler.milestones=[0] \
	callbacks.coarse2fine_scheduler.scales=[0.25] \
	# callbacks.coarse2fine_scheduler.milestones=[0,250,450,480] \
	# callbacks.coarse2fine_scheduler.scales=[0.05,0.1,0.25,0.5] \

point2point_flame:
	python scripts/optimize.py \
	model=point2point \
	data=flame \
	logger=wandb \
	model.lr=1e-02 \
	model.init_mode=flame \
	model.vertices_mask=full \
	trainer.max_epochs=50 \
	trainer.accelerator=gpu \
	callbacks.coarse2fine_scheduler.milestones=[0] \
	callbacks.coarse2fine_scheduler.scales=[0.25] \

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


