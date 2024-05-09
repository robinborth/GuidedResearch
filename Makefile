point2point:
	python scripts/optimize.py \
	model=point2point \
	data=kinect \
	logger=wandb \
	trainer.max_epochs=300 \
	callbacks.coarse2fine_scheduler.milestones=[0,150,250,280] \
	callbacks.coarse2fine_scheduler.image_scales=[0.05,0.1,0.25,0.5] \