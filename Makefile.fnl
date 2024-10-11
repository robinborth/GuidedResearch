##########################################################################
# make all -f Makefile.fnl
# GROUP0: make train -f Makefile.fnl
##########################################################################

.PHONY: all train__7-0E-01 train__5-0E-01 train__3-0E-01
all: train__7-0E-01 train__5-0E-01 train__3-0E-01

##########################################################################
# train
##########################################################################


train: train__7-0E-01 train__5-0E-01 train__3-0E-01

train__7-0E-01:
	python scripts/train.py \
	logger.group=train \
	logger.name=train__7-0E-01 \
	logger.tags=[train,train__7-0E-01] \
	task_name=train__7-0E-01 \
	data=kinect \
	framework.max_iters=2 \
	framework.max_optims=1 \
	trainer.max_epochs=50 \
	optimizer.step_size=0.7 \

train__5-0E-01:
	python scripts/train.py \
	logger.group=train \
	logger.name=train__5-0E-01 \
	logger.tags=[train,train__5-0E-01] \
	task_name=train__5-0E-01 \
	data=kinect \
	framework.max_iters=2 \
	framework.max_optims=1 \
	trainer.max_epochs=50 \
	optimizer.step_size=0.5 \

train__3-0E-01:
	python scripts/train.py \
	logger.group=train \
	logger.name=train__3-0E-01 \
	logger.tags=[train,train__3-0E-01] \
	task_name=train__3-0E-01 \
	data=kinect \
	framework.max_iters=2 \
	framework.max_optims=1 \
	trainer.max_epochs=50 \
	optimizer.step_size=0.3 \


