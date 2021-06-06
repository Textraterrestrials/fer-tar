#!/bin/bash
# samo skriptica da mi pokrene sve za redom

HIDDEN_SIZE=300
NUM_LAYERS=2
LR=5e-4
EPOCHS=10
DROPOUT=0.8
WEIGHT_DECAY=0.01
CLIP=0.5

# GPT2_5000_data_final
python3 main.py --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --lr $LR --clip $CLIP --epochs $EPOCHS --save_best --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
--model_name bilstm_gpt2_5000 --data_file ../../Training/Gudi_GPT_5000_data_final.csv --answers_file ../../Training/Gudi_GPT_5000_answers_final.csv

# GPT2_5000_data_final_log
# python3 main.py --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --lr $LR --clip $CLIP --epochs $EPOCHS --save_best --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
# --model_name bilstm_gpt2_log_5000 --data_file ../../Training/Gudi_GPT_5000_data_final_log.csv --answers_file ../../Training/Gudi_GPT_5000_answers_final_log.csv

# GPT2_5000_data_final_prod
python3 main.py --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --lr $LR --clip $CLIP --epochs $EPOCHS --save_best --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
--model_name bilstm_gtp2_prod_5000 --data_file ../../Training/Gudi_GPT_5000_data_final_prod.csv --answers_file ../../Training/Gudi_GPT_5000_answers_final_prod.csv

# backtranslation_5000
python3 main.py --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --lr $LR --clip $CLIP --epochs $EPOCHS --save_best --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
--model_name bilstm_bt_5000 --data_file ../../Training/Gudi_backtranslation_5000_data.csv --answers_file ../../Training/Gudi_backtranslation_5000_answers.csv

# plain data
python3 main.py --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --lr $LR --clip $CLIP --epochs $EPOCHS --save_best --dropout $DROPOUT --weight_decay $WEIGHT_DECAY \
--model_name bilstm_plain --data_file ../../Training/subtaskA_data_all.csv --answers_file ../../Training/subtaskA_answers_all.csv
