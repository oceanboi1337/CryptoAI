import requests 
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re, datetime

def _request(url: str, timeout: float = 5, retries: int = 3) -> requests.Response:
    for retry in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if not r.ok:
                continue
            return r
        except Exception as e:
            print(f'Attempt [{retry + 1}], error while fetching bitinfo data: {e}')

def _parse(str: str):
    str = re.sub('[\[\],\s]', str)
    return [x for x in re.split('[\'\"]', str) if x != '']

def fetch(start_date, end_date, charts: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date)})

    for chart in charts:
        data = _request(f'https://bitinfocharts.com/comparison/{chart}.html#alltime')
        if not data:
            continue

        soup = BeautifulSoup(data.text, 'html.parser')

        rows = []

        for script in soup.find_all('script'):
            if 'new Dygraph' in script.text:
                junk = ['[\'new Date("', '")', ']', '[new Date("', 'new Date("']
                rows = script.text.split('[[')[-1].split(']]')[0]

                for x in junk:
                    rows = rows.replace(x, '')

                rows = rows.split(',')

        if not rows:
            continue

        dates, values = np.array(rows[::2]), np.array(rows[1::2])
        values = [value.astype(np.float_) if value != 'null' else 0 for value in values]

        col = ''
        if 'hashrate' in chart: col = 'hashrate'
        elif 'tweets' in chart: col = 'tweets'

        tmp = pd.DataFrame({'date': pd.date_range(start=str(dates[0]), end=str(dates[-1]), periods=len(values)), col: values})
        df = pd.merge(df, tmp, on='date', how='outer')

        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        df.fillna(0, inplace=True)

        df.to_csv('datasets/bitinfo.csv')

        return df

if __name__ == '__main__':
    fetch('2014-09-17', '2024-02-22', ['bitcoin-hashrate', 'tweets-btc'])