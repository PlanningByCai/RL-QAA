from test.test_qaa import test_qaa
from test.test_qaoa import test_qaoa
import os

# 저장할 파일 경로 설정
save_dir = 'final_result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

episode = 50
epoch = 100
matrix_size= 9
hamming_weight =5

for matrix_idx in range(511, 520):
    test_qaa(
        num_episode=episode,
        num_epoch=epoch,
        beta=25.0,
        lr=0.5,
        matrix_size=matrix_size,
        matrix_idx=matrix_idx,
        hamming_weight=hamming_weight,
        model_name="RL_QAA",
        save_dir=save_dir,
    )

    test_qaa(
        num_episode=1,
        num_epoch=1,
        beta=100000000.0,
        lr=0.1,
        matrix_size=matrix_size,
        matrix_idx=matrix_idx,
        hamming_weight=hamming_weight,
        model_name="R_QAA",
        save_dir=save_dir,
    )

    test_qaoa(
        num_episode=episode,
        num_epoch=epoch,
        beta=25.0,
        lr=[0.1, 0.1],
        matrix_size=matrix_size,
        matrix_idx=matrix_idx,
        hamming_weight=hamming_weight,
        model_name="RL_QAOA",
        save_dir=save_dir,
    )
