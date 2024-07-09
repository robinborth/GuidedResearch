##########################################################################
# make lm_reg_1_0E-06 lm_reg_2_0E-05 lm_reg_5_0E-05 lm_reg_8_0E-05 lm_reg_1_0E-04 lm_reg_3_0E-04 lm_reg_5_0E-04 lm_reg_8_0E-04 lm_reg_1_0E-03 lm_reg_4_0E-03 lm_reg_7_0E-03 lm_reg_9_0E-03 lm_reg_2_0E-02 lm_reg_5_0E-02 lm_reg_8_0E-02 lm_reg_1_0E-02 lm_reg_4_0E-01 lm_reg_8_0E-01 lm_reg_1_0E+00 lm_test_1_0E-04 lm_test_2_0E-02
# GROUP0: make lm_reg_1_0E-06 lm_reg_2_0E-05 lm_reg_5_0E-05 lm_reg_8_0E-05 lm_reg_1_0E-04 lm_reg_3_0E-04 lm_reg_5_0E-04 lm_reg_8_0E-04 lm_reg_1_0E-03 lm_reg_4_0E-03 lm_reg_7_0E-03 lm_reg_9_0E-03 lm_reg_2_0E-02 lm_reg_5_0E-02 lm_reg_8_0E-02 lm_reg_1_0E-02 lm_reg_4_0E-01 lm_reg_8_0E-01 lm_reg_1_0E+00
# GROUP1: make lm_test_1_0E-04 lm_test_2_0E-02
##########################################################################


##########################################################################
# lm_reg
# make lm_reg_1_0E-06 lm_reg_2_0E-05 lm_reg_5_0E-05 lm_reg_8_0E-05 lm_reg_1_0E-04 lm_reg_3_0E-04 lm_reg_5_0E-04 lm_reg_8_0E-04 lm_reg_1_0E-03 lm_reg_4_0E-03 lm_reg_7_0E-03 lm_reg_9_0E-03 lm_reg_2_0E-02 lm_reg_5_0E-02 lm_reg_8_0E-02 lm_reg_1_0E-02 lm_reg_4_0E-01 lm_reg_8_0E-01 lm_reg_1_0E+00
##########################################################################

.PHONY: all lm_reg_1_0E lm_reg_2_0E

all: lm_reg_1_0E-06

lm_reg_1_0E-06:
	python scripts/optimize.py
	task_name=lm_reg_1_0E-06
	loss=regularization
	loss.chain.shape_regularization=1e-06
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_2_0E-05:
	python scripts/optimize.py
	task_name=lm_reg_2_0E-05
	loss=regularization
	loss.chain.shape_regularization=2e-05
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_5_0E-05:
	python scripts/optimize.py
	task_name=lm_reg_5_0E-05
	loss=regularization
	loss.chain.shape_regularization=5e-05
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_8_0E-05:
	python scripts/optimize.py
	task_name=lm_reg_8_0E-05
	loss=regularization
	loss.chain.shape_regularization=8e-05
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_1_0E-04:
	python scripts/optimize.py
	task_name=lm_reg_1_0E-04
	loss=regularization
	loss.chain.shape_regularization=0.0001
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_3_0E-04:
	python scripts/optimize.py
	task_name=lm_reg_3_0E-04
	loss=regularization
	loss.chain.shape_regularization=0.0003
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_5_0E-04:
	python scripts/optimize.py
	task_name=lm_reg_5_0E-04
	loss=regularization
	loss.chain.shape_regularization=0.0005
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_8_0E-04:
	python scripts/optimize.py
	task_name=lm_reg_8_0E-04
	loss=regularization
	loss.chain.shape_regularization=0.0008
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_1_0E-03:
	python scripts/optimize.py
	task_name=lm_reg_1_0E-03
	loss=regularization
	loss.chain.shape_regularization=0.001
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_4_0E-03:
	python scripts/optimize.py
	task_name=lm_reg_4_0E-03
	loss=regularization
	loss.chain.shape_regularization=0.004
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_7_0E-03:
	python scripts/optimize.py
	task_name=lm_reg_7_0E-03
	loss=regularization
	loss.chain.shape_regularization=0.007
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_9_0E-03:
	python scripts/optimize.py
	task_name=lm_reg_9_0E-03
	loss=regularization
	loss.chain.shape_regularization=0.009
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_2_0E-02:
	python scripts/optimize.py
	task_name=lm_reg_2_0E-02
	loss=regularization
	loss.chain.shape_regularization=0.02
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_5_0E-02:
	python scripts/optimize.py
	task_name=lm_reg_5_0E-02
	loss=regularization
	loss.chain.shape_regularization=0.05
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_8_0E-02:
	python scripts/optimize.py
	task_name=lm_reg_8_0E-02
	loss=regularization
	loss.chain.shape_regularization=0.08
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_1_0E-02:
	python scripts/optimize.py
	task_name=lm_reg_1_0E-02
	loss=regularization
	loss.chain.shape_regularization=0.01
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_4_0E-01:
	python scripts/optimize.py
	task_name=lm_reg_4_0E-01
	loss=regularization
	loss.chain.shape_regularization=0.4
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_8_0E-01:
	python scripts/optimize.py
	task_name=lm_reg_8_0E-01
	loss=regularization
	loss.chain.shape_regularization=0.8
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_reg_1_0E+00:
	python scripts/optimize.py
	task_name=lm_reg_1_0E+00
	loss=regularization
	loss.chain.shape_regularization=1.0
	loss.chain.expression_regularization=4e-04
	optimizer=levenberg_marquardt
	optimizer.verbose=False

##########################################################################
# lm_test
# make lm_test_1_0E-04 lm_test_2_0E-02
##########################################################################

lm_test_1_0E-04:
	python scripts/optimize.py
	task_name=lm_test_1_0E-04
	loss=regularization
	loss.chain.shape_regularization=1e-03
	loss.chain.expression_regularization=0.0001
	optimizer=levenberg_marquardt
	optimizer.verbose=False

lm_test_2_0E-02:
	python scripts/optimize.py
	task_name=lm_test_2_0E-02
	loss=regularization
	loss.chain.shape_regularization=1e-03
	loss.chain.expression_regularization=0.02
	optimizer=levenberg_marquardt
	optimizer.verbose=False

