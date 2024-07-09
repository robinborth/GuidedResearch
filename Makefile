####################################################################################
# PCG Sampling
####################################################################################

pcg_sampling:
	python scripts/pcg_sampling.py \
	task_name=pcg_sampling \
	optimizer=gauss_newton \
	loss=point2plane \
	pcg_sampling_trainer.max_samplings=1000 \
	pcg_sampling_trainer.max_iters=1 \
	pcg_sampling_trainer.max_optims=1 \


####################################################################################
# PCG Sampling
####################################################################################

# gauss_newton:
# 	python scripts/optimize.py \
# 	task_name=gauss_newton \
# 	optimizer=gauss_newton \
# 	optimizer.optimizer_params.pcg_steps=1 \
# 	optimizer.optimizer_params.pcg_jacobi=False \
# 	loss=point2plane \
# 	joint_trainer.final_video=False \
# 	joint_trainer.init_idxs=[0] \
# 	joint_trainer.max_iters=1 \
# 	joint_trainer.max_optims=1 \
# 	joint_trainer.scheduler.milestones=[0] \
# 	joint_trainer.scheduler.params=[[global_pose,transl]] \
# 	joint_trainer.coarse2fine.milestones=[0] \
# 	joint_trainer.coarse2fine.scales=[8] \
# 	sequential_trainer=null \

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
