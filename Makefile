levenberg_marquardt:
	python scripts/optimize.py \
	tags=["levenberg_marquardt"] \
	trainer.optimizer=levenberg_marquardt \
	trainer.optimizer_params={lin_solver:pytorch} \
	trainer.copy_optimizer_state=False \
	joint_trainer.max_iters=100 \
	joint_trainer.max_optims=5 \
	joint_trainer.optimizer.milestones=[0,7,10,12] \
	joint_trainer.optimizer.params=[[global_pose,transl],[neck_pose,eye_pose],[shape_params],[expression_params]] \
	sequential_trainer.max_iters=35 \
	sequential_trainer.max_optims=5 \
	sequential_trainer.optimizer.milestones=[0,3] \
	sequential_trainer.optimizer.params=[[global_pose,transl,neck_pose,eye_pose],[expression_params]] \

adam:
	python scripts/optimize.py \
	tags=["adam"] \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=True \
	joint_trainer.max_iters=20 \
	joint_trainer.max_optims=200 \
	joint_trainer.optimizer.milestones=[0,7,10] \
	joint_trainer.optimizer.params=[[global_pose,transl],[neck_pose,eye_pose],[shape_params,expression_params]] \
	sequential_trainer.max_iters=35 \
	sequential_trainer.max_optims=50 \
	sequential_trainer.optimizer.milestones=[0,3] \
	sequential_trainer.optimizer.params=[[global_pose,transl,neck_pose,eye_pose],[expression_params]] \

adam_low_lr:
	python scripts/optimize.py \
	tags=["adam_low_lr"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=False \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[1e-03,1e-03],[1e-03],[1e-03,1e-03]]

adam_copy:
	python scripts/optimize.py \
	tags=["adam_copy"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=True \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[1e-02,1e-02],[1e-02],[1e-02,1e-02]]

adam_copy_high_lr:
	python scripts/optimize.py \
	tags=["adam_copy_high_lr"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=True \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[1,1],[1],[1,1]]

ternary_linesearch:
	python scripts/optimize.py \
	tags=["ternary_linesearch"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=ternary_linesearch \
	trainer.copy_optimizer_state=False \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[0,0],[0],[0,0]]

gradient_decent_momentum_copy:
	python scripts/optimize.py \
	tags=["gradient_decent_momentum_copy"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent_momentum \
	trainer.copy_optimizer_state=True \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[10,10],[10],[10,10]]

gradient_decent_momentum:
	python scripts/optimize.py \
	tags=["gradient_decent_momentum"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent_momentum \
	trainer.copy_optimizer_state=False \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[10,10],[10],[10,10]]

gradient_decent:
	python scripts/optimize.py \
	tags=["gradient_decent"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent \
	trainer.copy_optimizer_state=False \
	trainer.coarse2fine.milestones=[0] \
	trainer.coarse2fine.scales=[8] \
	trainer.optimizer.milestones=[0,5,7] \
	trainer.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	trainer.optimizer.lr=[[10,10],[10],[10,10]]

create_video:
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_normal" +video_path="temp/render_normal.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/batch_color" +video_path="temp/batch_color.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/error_point_to_plane" +video_path="temp/error_point_to_plane.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_merged" +video_path="temp/render_merged.mp4"