import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_tf_forecasting import Exp_TF_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--t_model', type=str, default='PatchTST')
    parser.add_argument('--f_model', type=str, default='CompNet')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--exp_type', type=str, default='multi', help='experiment type')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--fnet_d_model', type=int, default=512, help='dimension of model of fnet')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--fnet_d_ff', type=int, default=2048, help='dimension of fcn of fnet')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--complex_dropout', type=float, default=0.1, help='complex_dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--device', type=str, default='cpu')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # additional params
    parser.add_argument('--low_pass', type=bool, default=False)  # 是否进行低通滤波
    parser.add_argument('--low_pass_threshold', type=float, default=3) # 低通滤波周期最低值
    parser.add_argument('--tf_loss', type=bool, default=False)
    parser.add_argument('--tf_loss_factor', type=float, default=0.5)

    args = parser.parse_args(args=[])
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_multi_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = '0,1,2,3'
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # settings
    args.task_name = 'long_term_forecast'
    args.is_training = 1
    args.root_path ='./datasets/'       
    args.exp_type = 'multi'
    args.checkpoints = './checkpoints/' + args.exp_type + '/'
    # args.data_path = 'solar.csv'
    # args.model_id = 'solar_336_720'
    # args.data_path = 'weather.csv'
    # args.model_id = 'weather_96_96'
    # args.data_path = 'national_illness.csv'
    # args.model_id = 'national_illness_36_36'
    # args.data_path = 'electricity.csv'
    # args.model_id = 'electricity_192_96'
    args.data_path = 'electricity_small.csv'
    args.model_id = 'electricity_small_192_96'
    # args.data_path = 'traffic.csv'
    # args.model_id = 'traffic_336_720'
    # args.data_path = 'ETTh1.csv'
    # args.model_id = 'etth1_96_96'
    # args.data_path = 'ETTh2.csv'
    # args.model_id = 'etth2_192_96'
    # args.data_path = 'faas/part-00000-b72e6747-b68c-4d59-81d7-d8f66e83598c-c000.gz.parquet'
    # args.model_id = 'faas_216_96'
    # args.data_path = 'hpa_csv/Split/periodic/Deployment_vke-system_vke-app-server_ap-johor_PROD-CTRL-JOHOR.csv'
    # args.model_id = 'hpa_cpuusage_420_60'
    # args.data_path = 'selected_abase.csv'
    # args.model_id = 'selected_abase_216_96'
    args.model = 'PatchTST'
    args.t_model = 'PatchTST'
    args.f_model = 'CompCNNv3'
    args.data = 'custom'
    args.features = 'M'
    # args.d_model = 16
    # args.d_ff = 32
    args.fnet_d_model = 8
    args.fnet_d_ff = 32
    args.seq_len = 192
    args.label_len = 24
    args.pred_len = 96
    args.e_layers = 2
    args.d_layers = 1
    # args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.freq = 'h'
    args.des = 'Exp'
    args.itr = 1
    args.train_epochs = 5
    args.batch_size = 32
    args.dropout = 0.1
    args.complex_dropout = 0.1

    
    args.low_pass = False
    args.low_pass_threshold = 1.5
    args.tf_loss = False
    args.tf_loss_factor = 0.5

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'tf_forecast':
        Exp = Exp_TF_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
