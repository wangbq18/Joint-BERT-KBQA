export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=1
python main.py \
    --do_predict \
    --model_dir experiments/debug \
    --nega_num 8