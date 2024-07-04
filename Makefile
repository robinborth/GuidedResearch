####################################################################################
# Different LM Linear Systems
####################################################################################

lm_JTF:
	python scripts/optimize.py \
	task_name=lm_JTF \
	loss=point2plane \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \

lm_JTF_reg:
	python scripts/optimize.py \
	task_name=lm_JTF_reg \
	loss=regularization \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=False \
	optimizer.verbose=False \

lm_grad:
	python scripts/optimize.py \
	task_name=lm_grad \
	loss=point2plane \
	optimizer=levenberg_marquardt \
	optimizer.use_grad=True \
	optimizer.verbose=False \


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