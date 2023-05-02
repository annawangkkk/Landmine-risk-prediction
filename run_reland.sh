CUDA_VISIBLE_DEVICES=0,1 python main.py --timestamp 05012023181236 \
--municipio puerto --subset full --model TabCmpt --n_step 2 \
--objective irm 2>&1 | tee -a ./experiments/05012023181236/log.txt