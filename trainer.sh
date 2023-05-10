# python softsensor_model.py --learning_rate0 1e-6 --learning_rate1 1e-6 --learning_rate2 1e-6 \
# --learning_rate_total 1e-6 --num_of_hidden_layers 0 --n_hidden 1024 \
# --batch_size 16 --exp_id total --n_epoch0 1000 --n_epoch1 1000 --n_epoch2 1000 --n_epoch_total 1000

python softsensor_model.py --learning_rate0 1e-2 --learning_rate1 1e-2 --learning_rate2 1e-2 \
--learning_rate_total 1e-2 --num_of_hidden_layers 0 --n_hidden 1024 \
--batch_size 8 --exp_id without_BN --n_epoch0 150 --n_epoch1 100 --n_epoch2 100 --n_epoch_total 50
# python softsensor_model.py --learning_rate0 1e-5 --learning_rate1 1e-5 --learning_rate2 1e-6 \
# --learning_rate_total 1e-6 --num_of_hidden_layers 0 --n_hidden 1024 \
# --batch_size 8 --exp_id total --n_epoch0 1000 --n_epoch1 1000 --n_epoch2 1000 --n_epoch_total 1000