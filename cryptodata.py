import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class CryptoDataset:
    def __init__(self, path: str, src_features: list[str], tgt_features: list[str], feature_engineering: bool = True, batch_size: int = 1, shuffle: bool = False):
        self.path = path
        self.src_features = [feature.lower() for feature in src_features]
        self.tgt_features = [feature.lower() for feature in tgt_features]
        self.feature_engineering = feature_engineering
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._train_data = None
        self._test_data = None
        self._val_data = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.engineered_features = []

        self.src_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.tgt_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.data = pd.read_csv(path)
        self.data.columns = self.data.columns.str.lower()

    def _calculate_rsi(self, periods: int = 14):
        delta = self.data['close'].diff(1)

        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.ewm(com=periods - 1, min_periods=periods).mean()
        avg_loss = loss.ewm(com=periods - 1, min_periods=periods).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _feature_engineering(self) -> list[str]:
        ef = pd.DataFrame()
        if 'volume' in self.data:
            #ef['volume_change'] = self.data['volume'].diff()
            pass

        if 'close' in self.data:
            #ef['rsi'] = self._calculate_rsi()
            #ef['close_lag1'] = self.data['close'].shift(1)
            #ef['close_lag2'] = self.data['close'].shift(2)
            ef['EMA_10'] = self.data['close'].ewm(span=20, adjust=False).mean()
            ef['EMA_50'] = self.data['close'].ewm(span=100, adjust=False).mean()
            #ef['Pct_Change'] = self.data['close'].pct_change() * 100

        if 'date' in self.data:
            if not np.issubdtype(self.data['date'].dtype, np.datetime64):
                self.data['date'] = pd.to_datetime(self.data['date'])

            #ef['day_of_week_sin'] = np.sin(2 * np.pi * self.data['date'].dt.weekday / 7)
            #ef['day_of_week_cos'] = np.cos(2 * np.pi * self.data['date'].dt.weekday / 7)
            #ef['month_sin'] = np.sin(2 * np.pi * self.data['date'].dt.month / 12)
            #ef['month_cos'] = np.cos(2 * np.pi * self.data['date'].dt.month / 12)

        self.engineered_features.extend(ef.columns.tolist())
        self.data = pd.concat([self.data, ef], axis=1)

        # Write out for debugging
        self.data.to_csv('datasets/formated.csv')

    @property
    def features(self) -> tuple:
        return self.src_features + self.engineered_features, self.tgt_features + self.engineered_features

    @property
    def features_length(self) -> tuple:
        return len(self.features[0]), len(self.features[1])

    def decode(self, output: np.ndarray):
        if output.shape[-1] != self.src_scaler.scale_.shape[0]:
            tmp = np.zeros((output.shape[0] * output.shape[1], self.features_length[0]))
            tmp[:, 1] = output.reshape(-1)
            return self.src_scaler.inverse_transform(tmp)
        else:
            return self.src_scaler.inverse_transform(output.reshape(-1, output.shape[-1]))

    def _normalize(self):
        self.src_scaler.fit(self._train_data[self.features[0]])
        #self.tgt_scaler.fit(self._train_data[self.features[1]])
        self._train_data = self.src_scaler.transform(self._train_data[self.features[0]])
        self._test_data = self.src_scaler.transform(self._test_data[self.features[0]])
        self._val_data = self.src_scaler.transform(self._val_data[self.features[0]])

    def _src_tgt_pairing(self, data: np.ndarray, src_length: int, tgt_length: int, to_tensor: bool = False):
        src, tgt = [], []
        
        for i in range(len(data) - src_length - tgt_length + 1):
            src.append(data[i:(i + src_length)])
            tgt.append(data[(i + src_length):(i + src_length + tgt_length)])
        
        if to_tensor:
            return torch.tensor(np.array(src), dtype=torch.float32), torch.tensor(np.array(tgt), dtype=torch.float32)

        return np.array(src), np.array(tgt)
    
    def _split_data(self, test: float = 0.2, val: float = 0.1, random_state: int = None, shuffle: bool = False):
        self._train_data, temp_data = train_test_split(self.data, test_size=(test + val), random_state=random_state, shuffle=shuffle)
        test_ratio = test / (test + val)
        self._test_data, self._val_data = train_test_split(temp_data, test_size=(1 - test_ratio), random_state=random_state, shuffle=shuffle)

    def preprocess(self, input_sequence_length: int, output_sequence_length: int, test_split: float, val_split: float):
        if self.feature_engineering:
            self._feature_engineering()

        self.data.bfill(inplace=True)

        self._split_data(test_split, val_split)

        self._normalize()

        self._src_train_data, self._tgt_train_data = self._src_tgt_pairing(self._train_data, input_sequence_length, output_sequence_length, to_tensor=True)
        self._src_test_data, self._tgt_test_data = self._src_tgt_pairing(self._test_data, input_sequence_length, output_sequence_length, to_tensor=True)
        self._src_val_data, self._tgt_val_data = self._src_tgt_pairing(self._val_data, input_sequence_length, output_sequence_length, to_tensor=True)

        self.train_loader = DataLoader(TensorDataset(self._src_train_data, self._tgt_train_data), batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_loader = DataLoader(TensorDataset(self._src_test_data, self._tgt_test_data), batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(TensorDataset(self._src_val_data, self._tgt_val_data), batch_size=self.batch_size, shuffle=self.shuffle)

        return True