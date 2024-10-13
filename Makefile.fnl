##########################################################################
# make all -f Makefile.fnl
##########################################################################

.PHONY: all train_kinect train_kinect
all: train_kinect train_kinect

train_kinect:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect \
	logger.tags=[train,train_kinect] \
	task_name=train_kinect \
	data=kinect \
	framework.max_iters=2 \
	framework.max_optims=1 \
	trainer.max_epochs=500 \
	optimizer.step_size=0.7 \

train_synthetic:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_synthetic \
	logger.tags=[train,train_synthetic] \
	task_name=train_synthetic \
	data=kinect \
	framework.max_iters=2 \
	framework.max_optims=1 \
	trainer.max_epochs=1200 \
	optimizer.step_size=0.7 \
	ckpt_path=/home/borth/GuidedResearch/checkpoints/synthetic_lr/ours2.ckpt


