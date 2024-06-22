levenberg_marquardt:
	python scripts/optimize.py \
	tags=["levenberg_marquardt"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.optimizer=levenberg_marquardt \
	scheduler.optimizer.copy_optimizer_state=False \

adam:
	python scripts/optimize.py \
	tags=["adam"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=False \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[1e-02,1e-02],[1e-02],[1e-02,1e-02]]

adam_low_lr:
	python scripts/optimize.py \
	tags=["adam_low_lr"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=False \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[1e-03,1e-03],[1e-03],[1e-03,1e-03]]

adam_copy:
	python scripts/optimize.py \
	tags=["adam_copy"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=True \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[1e-02,1e-02],[1e-02],[1e-02,1e-02]]

adam_copy_high_lr:
	python scripts/optimize.py \
	tags=["adam_copy_high_lr"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=200 \
	trainer.optimizer=adam \
	trainer.copy_optimizer_state=True \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[1,1],[1],[1,1]]

ternary_linesearch:
	python scripts/optimize.py \
	tags=["ternary_linesearch"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=ternary_linesearch \
	trainer.copy_optimizer_state=False \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[0,0],[0],[0,0]]

gradient_decent_momentum_copy:
	python scripts/optimize.py \
	tags=["gradient_decent_momentum_copy"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent_momentum \
	trainer.copy_optimizer_state=True \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[10,10],[10],[10,10]]

gradient_decent_momentum:
	python scripts/optimize.py \
	tags=["gradient_decent_momentum"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent_momentum \
	trainer.copy_optimizer_state=False \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[10,10],[10],[10,10]]

gradient_decent:
	python scripts/optimize.py \
	tags=["gradient_decent"] \
	data.batch_size=10 \
	model.optimize_frames=10 \
	trainer.max_iters=20 \
	trainer.max_optims=50 \
	trainer.optimizer=gradient_decent \
	trainer.copy_optimizer_state=False \
	scheduler.coarse2fine.milestones=[0] \
	scheduler.coarse2fine.scales=[8] \
	scheduler.optimizer.milestones=[0,5,7] \
	scheduler.optimizer.params=[["global_pose","transl"],["neck_pose"],["shape_params","expression_params"]] \
	scheduler.optimizer.lr=[[10,10],[10],[10,10]]

create_video:
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_normal" +video_path="temp/render_normal.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/batch_color" +video_path="temp/batch_color.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/error_point_to_plane" +video_path="temp/error_point_to_plane.mp4"
	python scripts/create_video.py +framerate=20 +video_dir="/home/borth/GuidedResearch/logs/optimize/runs/2024-06-07_12-49-31/render_merged" +video_path="temp/render_merged.mp4"