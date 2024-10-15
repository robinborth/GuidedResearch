##########################################################################
# make all -f Makefile.fnl
##########################################################################

.PHONY: all train_kinect train_synthetic
all: train_kinect train_synthetic

train_kinect:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect \
	logger.tags=[train,train_kinect] \
	task_name=train_kinect \
	data=kinect \
	data.train_dataset.jump_size=4 \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=500 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \
	ckpt_path=/home/borth/GuidedResearch/logs/2024-10-14/19-05-04_train_kinect/checkpoints/epoch_399.ckpt \

train_kinect_wo_prior:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect_wo_prior \
	logger.tags=[train,train_kinect_wo_prior] \
	task_name=train_kinect_wo_prior \
	data=kinect \
	data.train_dataset.jump_size=4 \
	residuals=face2face_wo_landmarks \
	regularize=dummy \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=500 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \
	ckpt_path=/home/borth/GuidedResearch/logs/2024-10-14/19-05-04_train_kinect_wo_prior/checkpoints/epoch_439.ckpt \


train_synthetic:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_synthetic \
	logger.tags=[train,train_synthetic] \
	task_name=train_synthetic \
	data=kinect \
	data.train_dataset.jump_size=4 \
	framework.lr=1e-05 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=1200 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \
	ckpt_path=/home/borth/GuidedResearch/checkpoints/synthetic_lr/ours2.ckpt


