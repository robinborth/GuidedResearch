####################################################################################
# PCG Sampling
####################################################################################

pcg_sampling:
	python scripts/pcg_sampling.py \
	task_name=pcg_sampling \
	optimizer=gauss_newton \
	loss=point2plane \
	pcg_sampling_trainer.max_samplings=10000 \
	pcg_sampling_trainer.max_iters=5 \
	pcg_sampling_trainer.max_optims=3 \

pcg:
	python scripts/pcg_training.py \
	task_name=pcg \
	model.optimizer.lr=1e-06 \
	model.max_iter=1 \
	data.batch_size=1 \
	trainer.overfit_batches=1 \

####################################################################################
# Optimization
####################################################################################

levenberg-marquardt:
	python scripts/optimize.py \
	task_name=levenberg_marquardt \
	optimizer=levenberg_marquardt \
	optimizer.store_system=False \
	optimizer.verbose=False \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=7e-04 \
	joint_trainer.init_idxs=[0] \
	joint_trainer.verbose=False \
	sequential_trainer=null \

gauss_newton:
	python scripts/optimize.py \
	model=flame \
	task_name=gauss_newton \
	optimizer=gauss_newton \
	residuals=regularization \
	joint_tracker.max_iters=50 \
	joint_tracker.max_optims=5 \
	joint_tracker.init_idxs=[0] \
	sequential_tracker.max_iters=8 \
	sequential_tracker.max_optims=3 \
	sequential_tracker.end_frame=50 \

adam:
	python scripts/optimize.py \
	task_name=adam \
	optimizer=adam \
	loss=point2plane \
	joint_trainer.scheduler.copy_optimizer_state=True \
	joint_trainer.scheduler.milestones=[0,10,20] \
	joint_trainer.scheduler.params=[[global_pose,transl],[neck_pose,eye_pose],[shape_params,expression_params]] \
	joint_trainer.max_iters=100 \
	joint_trainer.max_optims=10 \
	joint_trainer.init_idxs=[0] \
	joint_trainer.verbose=True \
	sequential_trainer=null \


landmark2d:
	python scripts/optimize.py \
	task_name=landmark2d \
	optimizer=gauss_newton \
	optimizer.step_size=1.0 \
	optimizer.store_system=False \
	optimizer.verbose=False \
	loss=landmark2d \
	joint_trainer.max_iters=200 \
	joint_trainer.max_optims=10 \
	joint_trainer.scheduler.milestones=[0,20,40] \
	joint_trainer.scheduler.params=[[global_pose,transl],[neck_pose],[shape_params,expression_params]] \
	sequential_trainer=null \

# optimizer.lin_solver._target_=lib.optimizer.pcg.PCGSolver \
# +optimizer.lin_solver.condition_net._target_=lib.optimizer.pcg.JaccobiConditionNet \
####################################################################################
# PCG Sampling
####################################################################################

lm:
	python scripts/optimize.py \
	task_name=levenberg_marquardt \
	optimizer=levenberg_marquardt \
	optimizer.store_system=True \
	loss=regularization \
	loss.chain.shape_regularization=4e-03 \
	loss.chain.expression_regularization=7e-04 \
	sequential_trainer=null \

####################################################################################
# Reg Tests 
####################################################################################

lm_reg_0:
	python scripts/optimize.py \
	task_name=lm_reg_0 \
	loss=regularization \
	loss.chain.shape_regularization=7e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_1:
	python scripts/optimize.py \
	task_name=lm_reg_1 \
	loss=regularization \
	loss.chain.shape_regularization=5e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_2:  # this is good
	python scripts/optimize.py \
	task_name=lm_reg_2 \
	loss=regularization \
	loss.chain.shape_regularization=4e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_3:
	python scripts/optimize.py \
	task_name=lm_reg_3 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_4:
	python scripts/optimize.py \
	task_name=lm_reg_4 \
	loss=regularization \
	loss.chain.shape_regularization=2e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_5:
	python scripts/optimize.py \
	task_name=lm_reg_5 \
	loss=regularization \
	loss.chain.shape_regularization=1e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_6:
	python scripts/optimize.py \
	task_name=lm_reg_6 \
	loss=regularization \
	loss.chain.shape_regularization=5e-03 \
	loss.chain.expression_regularization=1e-04 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_7:
	python scripts/optimize.py \
	task_name=lm_reg_7 \
	loss=regularization \
	loss.chain.shape_regularization=5e-03 \
	loss.chain.expression_regularization=1e-05 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_8:
	python scripts/optimize.py \
	task_name=lm_reg_8 \
	loss=regularization \
	loss.chain.shape_regularization=5e-03 \
	loss.chain.expression_regularization=1e-06 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_9:
	python scripts/optimize.py \
	task_name=lm_reg_9 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=1e-04 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_10:
	python scripts/optimize.py \
	task_name=lm_reg_10 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=1e-05 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_11:
	python scripts/optimize.py \
	task_name=lm_reg_11 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=1e-06 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \



####################################################################################
# Reg Tests Expression
####################################################################################

lm_reg_3_0: 
	python scripts/optimize.py \
	task_name=lm_reg_3_0 \
	loss=regularization \
	loss.chain.shape_regularization=4e-03 \
	loss.chain.expression_regularization=7e-04 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_3_1:
	python scripts/optimize.py \
	task_name=lm_reg_3_1 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=4e-04 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

lm_reg_3_2: 
	python scripts/optimize.py \
	task_name=lm_reg_3_2 \
	loss=regularization \
	loss.chain.shape_regularization=3e-03 \
	loss.chain.expression_regularization=1e-04 \
	optimizer=levenberg_marquardt \
	optimizer.verbose=False \

####################################################################################
# Reg Tests Expression
####################################################################################

pcg_lr: 
	python scripts/pcg_training.py --multirun \
	task_name=pcg_lr \
	model.optimizer.lr=1e-06,2e-05,5e-05,8e-05,1e-04,3e-04,5e-04,8e-04,1e-03,4e-03,7e-03,9e-03,2e-02,5e-02,8e-02,1e-02,4e-01,8e-01,1.0 \



pcg_scheduler:
	python scripts/pcg_training.py \
	logger.group=pcg_scheduler \
	logger.name=pcg_scheduler \
	logger.tags=[pcg_scheduler,pcg_scheduler] \
	task_name=pcg_scheduler \
	model.optimizer.lr=1e-03 \
	model.max_iter=5 \
	model.loss._target_=lib.optimizer.pcg.L1SolutionLoss \
	model.condition_net._target_=lib.optimizer.pcg.DenseConditionNet \
	model.condition_net.num_layers=2 \
	+model.scheduler._target_=torch.optim.lr_scheduler.ExponentialLR \
	+model.scheduler._partial_=True \
	+model.scheduler.gamma=0.999 \
	data.batch_size=1024 \
	trainer.max_epochs=10000 \



.PHONY: iters_optims iters4_optims1  iters5_optims1 iters1_optims4 iters1_optims5 
iters_optims: iters4_optims1  iters5_optims1 iters1_optims4 iters1_optims5

iters4_optims1:
	python scripts/training.py \
	logger.name=iters4_optims1 \
	logger.tags=[train,iters_optims] \
	framework.max_iters=4 \
	framework.max_optims=1 \

iters5_optims1:
	python scripts/training.py \
	logger.name=iters5_optims1 \
	logger.tags=[train,iters_optims] \
	framework.max_iters=5 \
	framework.max_optims=1 \

iters1_optims4:
	python scripts/training.py \
	logger.name=iters1_optims4 \
	logger.tags=[train,iters_optims] \
	framework.max_iters=1 \
	framework.max_optims=4 \

iters1_optims5:
	python scripts/training.py \
	logger.name=iters1_optims5 \
	logger.tags=[train,iters_optims] \
	framework.max_iters=1 \
	framework.max_optims=5 \