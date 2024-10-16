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



.PHONY: iters_optims iters1_optims1 iters2_optims1 iters3_optims1 iters4_optims1 iters5_optims1 iters1_optims2 iters1_optims3 iters1_optims4 iters1_optims5 
iters_optims: iters4_optims1 iters1_optims1 iters2_optims1 iters3_optims1 iters4_optims1 iters5_optims1 iters1_optims2 iters1_optims3 iters1_optims4 iters1_optims5 

iters1_optims1:
	python scripts/training.py \
	logger.name=iters1_optims1 \
	logger.tags=[train,iters_optims] \
	logger.group=base \
	framework.max_iters=1 \
	framework.max_optims=1 \

iters2_optims1:
	python scripts/training.py \
	logger.name=iters2_optims1 \
	logger.tags=[train,iters_optims] \
	logger.group=iters \
	framework.max_iters=2 \
	framework.max_optims=1 \

iters3_optims1:
	python scripts/training.py \
	logger.name=iters3_optims1 \
	logger.tags=[train,iters_optims] \
	logger.group=iters \
	framework.max_iters=3 \
	framework.max_optims=1 \

iters4_optims1:
	python scripts/training.py \
	logger.name=iters4_optims1 \
	logger.tags=[train,iters_optims] \
	logger.group=iters \
	framework.max_iters=4 \
	framework.max_optims=1 \

iters5_optims1:
	python scripts/training.py \
	logger.name=iters5_optims1 \
	logger.tags=[train,iters_optims] \
	logger.group=iters \
	framework.max_iters=5 \
	framework.max_optims=1 \

iters1_optims2:
	python scripts/training.py \
	logger.name=iters1_optims2 \
	logger.tags=[train,iters_optims] \
	logger.group=optims \
	framework.max_iters=1 \
	framework.max_optims=2 \

iters1_optims3:
	python scripts/training.py \
	logger.name=iters1_optims3 \
	logger.tags=[train,iters_optims] \
	logger.group=optims \
	framework.max_iters=1 \
	framework.max_optims=3 \

iters1_optims4:
	python scripts/training.py \
	logger.name=iters1_optims4 \
	logger.tags=[train,iters_optims] \
	logger.group=optims \
	framework.max_iters=1 \
	framework.max_optims=4 \

iters1_optims5:
	python scripts/training.py \
	logger.name=iters1_optims5 \
	logger.tags=[train,iters_optims] \
	logger.group=optims \
	framework.max_iters=1 \
	framework.max_optims=5 \

.PHONY: optimize dphm_christoph_mouthmove dphm_christoph_rotatemouth dphm_innocenzo_fulgintl_mouthmove dphm_innocenzo_fulgintl_rotatemouth
optimize: dphm_christoph_mouthmove dphm_christoph_rotatemouth dphm_innocenzo_fulgintl_mouthmove dphm_innocenzo_fulgintl_rotatemouth

dphm_christoph_mouthmove:
	python scripts/optimize.py \
	logger.name=dphm_christoph_mouthmove \
	logger.tags=[optimize,dphm_christoph_mouthmove] \
	data.dataset_name=dphm_christoph_mouthmove \

dphm_christoph_rotatemouth:
	python scripts/optimize.py \
	logger.name=dphm_christoph_rotatemouth \
	logger.tags=[optimize,dphm_christoph_rotatemouth] \
	data.dataset_name=dphm_christoph_rotatemouth \

dphm_innocenzo_fulgintl_mouthmove:
	python scripts/optimize.py \
	logger.name=dphm_innocenzo_fulgintl_mouthmove \
	logger.tags=[optimize,dphm_innocenzo_fulgintl_mouthmove] \
	data.dataset_name=dphm_innocenzo_fulgintl_mouthmove \

dphm_innocenzo_fulgintl_rotatemouth:
	python scripts/optimize.py \
	logger.name=dphm_innocenzo_fulgintl_rotatemouth \
	logger.tags=[optimize,dphm_innocenzo_fulgintl_rotatemouth] \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \

####################################################################################
# Masters Day Results
####################################################################################


.PHONY: results results_fulgitl1 results_fulgitl2 results_christoph1 results_christoph2
results: results_fulgitl1 results_fulgitl2 results_christoph1 results_christoph2


####################################################################################
# Masters Day Results (Fulgitl-1)
####################################################################################


.PHONY: results_fulgitl1 results_fulgitl1_1_params_p2p results_fulgitl1_3_params_p2p results_fulgitl1_3_params results_fulgitl1_3_p2p results_fulgitl1_10_params_p2p
results_fulgitl1: results_fulgitl1_1_params_p2p results_fulgitl1_3_params_p2p results_fulgitl1_3_params results_fulgitl1_3_p2p results_fulgitl1_10_params_p2p


results_fulgitl1_1_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl1_1_params_p2p \
	logger.tags=[train,results_fulgitl1_1_params_p2p] \
	logger.group=train \
	framework.max_iters=1 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=6 \
	data.train_dataset.start_frame=31 \
	framework.log_frame_idx=31 \

results_fulgitl1_3_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl1_3_params_p2p \
	logger.tags=[train,results_fulgitl1_3_params_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=6 \
	data.train_dataset.start_frame=31 \
	framework.log_frame_idx=31 \

results_fulgitl1_3_params:
	python scripts/training.py \
	logger.name=results_fulgitl1_3_params \
	logger.tags=[train,results_fulgitl1_3_params] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=6 \
	data.train_dataset.start_frame=31 \
	framework.log_frame_idx=31 \

results_fulgitl1_3_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl1_3_p2p \
	logger.tags=[train,results_fulgitl1_3_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=0.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=6 \
	data.train_dataset.start_frame=31 \
	framework.log_frame_idx=31 \

results_fulgitl1_10_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl1_10_params_p2p \
	logger.tags=[train,results_fulgitl1_10_params_p2p] \
	logger.group=train \
	framework.max_iters=10 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=6 \
	data.train_dataset.start_frame=31 \
	framework.log_frame_idx=31 \

####################################################################################
# Masters Day Results (Fulgitl-2)
####################################################################################


.PHONY: results_fulgitl2 results_fulgitl2_1_params_p2p results_fulgitl2_3_params_p2p results_fulgitl2_3_params results_fulgitl2_3_p2p results_fulgitl2_10_params_p2p
results_fulgitl2: results_fulgitl2_1_params_p2p results_fulgitl2_3_params_p2p results_fulgitl2_3_params results_fulgitl2_3_p2p results_fulgitl2_10_params_p2p


results_fulgitl2_1_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl2_1_params_p2p \
	logger.tags=[train,results_fulgitl2_1_params_p2p] \
	logger.group=train \
	framework.max_iters=1 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=2 \
	data.train_dataset.start_frame=47 \
	framework.log_frame_idx=47 \

results_fulgitl2_3_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl2_3_params_p2p \
	logger.tags=[train,results_fulgitl2_3_params_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=2 \
	data.train_dataset.start_frame=47 \
	framework.log_frame_idx=47 \

results_fulgitl2_3_params:
	python scripts/training.py \
	logger.name=results_fulgitl2_3_params \
	logger.tags=[train,results_fulgitl2_3_params] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=2 \
	data.train_dataset.start_frame=47 \
	framework.log_frame_idx=47 \

results_fulgitl2_3_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl2_3_p2p \
	logger.tags=[train,results_fulgitl2_3_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=0.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=2 \
	data.train_dataset.start_frame=47 \
	framework.log_frame_idx=47 \

results_fulgitl2_10_params_p2p:
	python scripts/training.py \
	logger.name=results_fulgitl2_10_params_p2p \
	logger.tags=[train,results_fulgitl2_10_params_p2p] \
	logger.group=train \
	framework.max_iters=10 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_innocenzo_fulgintl_rotatemouth \
	data.train_dataset.jump_size=2 \
	data.train_dataset.start_frame=47 \
	framework.log_frame_idx=47 \

####################################################################################
# Masters Day Results (Christoph-1)
####################################################################################

.PHONY: results_christoph1 results_christoph1_1_params_p2p results_christoph1_3_params_p2p results_christoph1_3_params results_christoph1_3_p2p results_christoph1_10_params_p2p
results_christoph1: results_christoph1_1_params_p2p results_christoph1_3_params_p2p results_christoph1_3_params results_christoph1_3_p2p results_christoph1_10_params_p2p


results_christoph1_1_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph1_1_params_p2p \
	logger.tags=[train,results_christoph1_1_params_p2p] \
	logger.group=train \
	framework.max_iters=1 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=52 \
	framework.log_frame_idx=52 \

results_christoph1_3_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph1_3_params_p2p \
	logger.tags=[train,results_christoph1_3_params_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=52 \
	framework.log_frame_idx=52 \

results_christoph1_3_params:
	python scripts/training.py \
	logger.name=results_christoph1_3_params \
	logger.tags=[train,results_christoph1_3_params] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=52 \
	framework.log_frame_idx=52 \

results_christoph1_3_p2p:
	python scripts/training.py \
	logger.name=results_christoph1_3_p2p \
	logger.tags=[train,results_christoph1_3_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=0.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=52 \
	framework.log_frame_idx=52 \

results_christoph1_10_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph1_10_params_p2p \
	logger.tags=[train,results_christoph1_10_params_p2p] \
	logger.group=train \
	framework.max_iters=10 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=52 \
	framework.log_frame_idx=52 \


####################################################################################
# Masters Day Results (Christoph-2)
####################################################################################

.PHONY: results_christoph2 results_christoph2_1_params_p2p results_christoph2_3_params_p2p results_christoph2_3_params results_christoph2_3_p2p results_christoph2_10_params_p2p
results_christoph2: results_christoph2_1_params_p2p results_christoph2_3_params_p2p results_christoph2_3_params results_christoph2_3_p2p results_christoph2_10_params_p2p


results_christoph2_1_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph2_1_params_p2p \
	logger.tags=[train,results_christoph2_1_params_p2p] \
	logger.group=train \
	framework.max_iters=1 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=42 \
	framework.log_frame_idx=42 \

results_christoph2_3_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph2_3_params_p2p \
	logger.tags=[train,results_christoph2_3_params_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=42 \
	framework.log_frame_idx=42 \

results_christoph2_3_params:
	python scripts/training.py \
	logger.name=results_christoph2_3_params \
	logger.tags=[train,results_christoph2_3_params] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=42 \
	framework.log_frame_idx=42 \

results_christoph2_3_p2p:
	python scripts/training.py \
	logger.name=results_christoph2_3_p2p \
	logger.tags=[train,results_christoph2_3_p2p] \
	logger.group=train \
	framework.max_iters=3 \
	framework.max_optims=1 \
	framework.param_weight=0.0 \
	framework.geometric_weight=0.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=42 \
	framework.log_frame_idx=42 \

results_christoph2_10_params_p2p:
	python scripts/training.py \
	logger.name=results_christoph2_10_params_p2p \
	logger.tags=[train,results_christoph2_10_params_p2p] \
	logger.group=train \
	framework.max_iters=10 \
	framework.max_optims=1 \
	framework.param_weight=1.0 \
	framework.geometric_weight=1.0 \
	data.dataset_name=dphm_christoph_mouthmove \
	data.train_dataset.jump_size=4 \
	data.train_dataset.start_frame=42 \
	framework.log_frame_idx=42 \


# innocenzo_fulgintl_rotatemouth, christoph_mouthmove, ali_kocal_mouthmove 
tracking:
	python scripts/optimize.py \
	optimizer=gauss_newton \
	optimizer.step_size=3e-01 \
	correspondence=projective \
	correspondence.d_threshold=0.02 \
	correspondence.n_threshold=0.95 \
	residuals=face2face \
	residuals.chain.shape_regularization.weight=7e-03 \
	residuals.chain.expression_regularization.weight=1e-03 \
	residuals.chain.neck_regularization.weight=5e-02 \
	joint_tracker.max_iters=150 \
	joint_tracker.max_optims=1 \
	joint_tracker.coarse2fine.milestones=[0,70] \
	joint_tracker.coarse2fine.scales=[8,4] \
	joint_tracker.save_interval=10 \
	sequential_tracker.max_iters=50 \
	sequential_tracker.max_optims=1 \
	sequential_tracker.coarse2fine.milestones=[0,20] \
	sequential_tracker.coarse2fine.scales=[8,4] \
	sequential_tracker.save_interval=10 \
	data.data_dir=/home/borth/GuidedResearch/data/debug \
	data.dataset_name=christoph_mouthmove \

tracking2:
	python scripts/optimize.py \
	optimizer=gauss_newton \
	optimizer.step_size=3e-01 \
	correspondence=projective \
	correspondence.d_threshold=0.02 \
	correspondence.n_threshold=0.95 \
	residuals=face2face \
	residuals.chain.shape_regularization.weight=5e-03 \
	residuals.chain.expression_regularization.weight=2e-03 \
	residuals.chain.neck_regularization.weight=1e-02 \
	joint_tracker.max_iters=150 \
	joint_tracker.max_optims=1 \
	joint_tracker.coarse2fine.milestones=[0,70] \
	joint_tracker.coarse2fine.scales=[8,4] \
	joint_tracker.save_interval=10 \
	sequential_tracker.max_iters=50 \
	sequential_tracker.max_optims=1 \
	sequential_tracker.coarse2fine.milestones=[0,20] \
	sequential_tracker.coarse2fine.scales=[8,4] \
	sequential_tracker.save_interval=10 \
	data.data_dir=/home/borth/GuidedResearch/data/debug \
	data.dataset_name=christoph_mouthmove \


tracking3:
	python scripts/optimize.py \
	optimizer=gauss_newton \
	optimizer.step_size=3e-01 \
	correspondence=projective \
	correspondence.d_threshold=0.02 \
	correspondence.n_threshold=0.95 \
	residuals=face2face \
	residuals.chain.shape_regularization.weight=5e-03 \
	residuals.chain.expression_regularization.weight=2e-03 \
	residuals.chain.neck_regularization.weight=1e-03 \
	joint_tracker.max_iters=150 \
	joint_tracker.max_optims=1 \
	joint_tracker.coarse2fine.milestones=[0,70] \
	joint_tracker.coarse2fine.scales=[8,4] \
	joint_tracker.save_interval=10 \
	sequential_tracker.max_iters=50 \
	sequential_tracker.max_optims=1 \
	sequential_tracker.coarse2fine.milestones=[0,20] \
	sequential_tracker.coarse2fine.scales=[8,4] \
	sequential_tracker.save_interval=10 \
	data.data_dir=/home/borth/GuidedResearch/data/debug \
	data.dataset_name=christoph_mouthmove \

