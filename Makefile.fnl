train_kinect:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect \
	logger.tags=[train,train_kinect] \
	task_name=train_kinect \
    data=kinect \
    framework.max_iters=3 \
    framework.max_optims=1 \
    trainer.max_epochs=100 \

train_syn:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_syn \
	logger.tags=[train,train_syn] \
	task_name=train_syn \
    data=kinect \
    framework.max_iters=3 \
    framework.max_optims=1 \
    trainer.max_epochs=1100 \
    ckpt_path=/home/borth/GuidedResearch/checkpoints/synthetic/ours.ckpt \
