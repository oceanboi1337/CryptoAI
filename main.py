import torch
import cryptodata
from sklearn.preprocessing import RobustScaler
from metrics import ModelMetrics
from model import Transformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64

    src_features = ['open', 'high', 'low', 'close', 'volume']
    #src_features = ['close']


    dataset = cryptodata.CryptoDataset('datasets/BTC-USD-daily.csv', src_features, src_features, feature_engineering=True, batch_size=batch_size, shuffle=False)

    # Use the last 30 days to predict the next 7 days
    previous_history = 60
    future_length = 1

    if not dataset.preprocess(previous_history, future_length, test_split=0.2, val_split=0.1):
        print('Faled to preprocess dataset')
        return

    input_dim, output_dim = dataset.features_length
    model_dim = 512
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 2048
    dropout = 0.3
    output_seq_len = 1
    lr = 0.0001

    model = Transformer(input_dim, output_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, lr, output_seq_len, device=device, background=True)
    model = model.to(device)

    model_metrics = ModelMetrics(dataset)

    for epoch, loss in model.train_model(dataset.train_loader, epochs=1000, verbose=True):
        if epoch % 2 == 0:
            for eval_loss, output, target in model.evaluate(dataset.val_loader, auto_regression=True):
                model_metrics.plot(output, target)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopping')