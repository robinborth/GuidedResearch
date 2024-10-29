##########################################################################
# make all -f Makefile.fnl
##########################################################################

.PHONY: all train_kinect train_synthetic
all: train_kinect train_synthetic

train_kinect_fix:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect_fix \
	logger.tags=[train,train_kinect_fix] \
	task_name=train_kinect_fix \
	data=kinect \
	data.train_dataset.jump_size=4 \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.00 \
	framework.vertices_weight=0.1 \
	trainer.max_epochs=500 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \

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
	framework.residual_weight=0.0 \
	framework.vertices_weight=0.1 \
	trainer.max_epochs=1200 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \
	ckpt_path=/home/borth/GuidedResearch/checkpoints/synthetic_lr/ours2.ckpt

train_kinect_f2f:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect_f2f \
	logger.tags=[train,train_kinect_f2f] \
	task_name=train_kinect_f2f \
	data=kinect \
	data.train_dataset.jump_size=4 \
	data.scale=4 \
	weighting.size=512 \
	residuals=face2face \
	regularize=dummy \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=1000 \
	optimizer.step_size=0.3 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \

train_kinect_neural_w_lm:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect_neural_w_lm \
	logger.tags=[train,train_kinect_neural_w_lm] \
	task_name=train_kinect_neural_w_lm \
	data=kinect \
	data.train_dataset.jump_size=4 \
	data.scale=4 \
	weighting.size=512 \
	residuals=face2face \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=1000 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \

train_kinect_neural_w_lm_wo_prior:
	python scripts/train.py \
	logger.group=train \
	logger.name=train_kinect_neural_w_lm_wo_prior \
	logger.tags=[train,train_kinect_neural_w_lm_wo_prior] \
	task_name=train_kinect_neural_w_lm_wo_prior \
	data=kinect \
	data.train_dataset.jump_size=4 \
	data.scale=4 \
	weighting.size=512 \
	residuals=neural_w_landmarks \
	regularize=dummy \
	framework.lr=1e-04 \
	framework.max_iters=2 \
	framework.max_optims=1 \
	framework.residual_weight=0.05 \
	framework.vertices_weight=0.03 \
	trainer.max_epochs=1000 \
	optimizer.step_size=0.7 \
	optimizer.lin_solver._target_=lib.optimizer.solver.PytorchEpsSolver \
			