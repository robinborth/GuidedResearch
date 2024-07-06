####################################################################################
# Different LM Linear Systems
####################################################################################

lm:
	python scripts/optimize.py \
	task_name=levenberg_marquardt \
	loss=point2plane \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_reg_shape_23_expr_8:
	python scripts/optimize.py \
	task_name=lm_reg_shape_23_expr_8 \
	loss=regularization \
	loss.chain.shape_regularization=5e-03 \
	loss.chain.expression_regularization=1e-08 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_reg_shape_3_expr_3:
	python scripts/optimize.py \
	task_name=lm_reg_shape_3_expr_3 \
	loss=regularization \
	loss.chain.shape_regularization=1e-03 \
	loss.chain.expression_regularization=1e-03 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_reg_shape_2_expr_2:
	python scripts/optimize.py \
	task_name=lm_reg_shape_2_expr_2 \
	loss=regularization \
	loss.chain.shape_regularization=1e-02 \
	loss.chain.expression_regularization=1e-02 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_reg_6:
	python scripts/optimize.py \
	task_name=lm_reg_6 \
	loss=regularization \
	loss.chain.regularization=1e-05 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_JTF_reg_7:
	python scripts/optimize.py \
	task_name=lm_JTF_reg_7 \
	loss=regularization \
	loss.chain.regularization=1e-07 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \
	sequential_trainer=null \

lm_grad:
	python scripts/optimize.py \
	task_name=lm_grad \
	loss=point2plane \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=True \
	optimizer.verbose=False \
	sequential_trainer=null \

# now we need more regularization!
lm_grad_reg_6:
	python scripts/optimize.py \
	task_name=lm_grad \
	loss=regularization \
	loss.chain.regularization=1e-06 \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=True \
	optimizer.verbose=False \
	sequential_trainer=null \


####################################################################################
# PCG Sampling
####################################################################################


pcg_sampling_graph:
	python scripts/pcg_sampling.py \
	task_name=pcg_sampling \
	optimizer=gauss_newton \
	loss=point2plane \
	max_samplings=100 \
	joint_trainer.verbose=True \
	joint_trainer.init_idxs=[0] \
	joint_trainer.max_iters=1 \
	joint_trainer.max_optims=5 \
	joint_trainer.scheduler.milestones=[0] \
	joint_trainer.scheduler.params=[[global_pose,transl]] \
	joint_trainer.coarse2fine.milestones=[0] \
	joint_trainer.coarse2fine.scales=[8] \
	sequential_trainer=null \

pcg_sampling:
	python scripts/pcg_sampling.py \
	task_name=pcg_sampling \
	optimizer=gauss_newton \
	loss=point2plane \
	max_samplings=2000 \
	joint_trainer.init_idxs=[0] \
	joint_trainer.max_iters=1 \
	joint_trainer.max_optims=5 \
	joint_trainer.scheduler.milestones=[0] \
	joint_trainer.scheduler.params=[[global_pose,transl]] \
	joint_trainer.coarse2fine.milestones=[0] \
	joint_trainer.coarse2fine.scales=[8] \
	sequential_trainer=null \


####################################################################################
# PCG Sampling
####################################################################################

gauss_newton:
	python scripts/optimize.py \
	task_name=gauss_newton \
	optimizer=gauss_newton \
	optimizer.optimizer_params.pcg_steps=1 \
	optimizer.optimizer_params.pcg_jacobi=False \
	loss=point2plane \
	joint_trainer.final_video=False \
	joint_trainer.init_idxs=[0] \
	joint_trainer.max_iters=1 \
	joint_trainer.max_optims=1 \
	joint_trainer.scheduler.milestones=[0] \
	joint_trainer.scheduler.params=[[global_pose,transl]] \
	joint_trainer.coarse2fine.milestones=[0] \
	joint_trainer.coarse2fine.scales=[8] \
	sequential_trainer=null \