"""
╔══════════════════════════════════════════════════════════════════╗
║              CIPHER V3 — Smart Futures Trading Bot              ║
║                                                                  ║
║  YENİ: • 15 Majör Coin (LONG + SHORT)                           ║
║        • Kademeli Giriş (25 + 75 USDT)                         ║
║        • Destek/Direnç Bazlı Dinamik SL                         ║
║        • Swing High/Low + Pivot + Order Block                   ║
║        • HH/HL Yapısal Trend Tespiti                            ║
║        • Taker Buy/Sell Ratio                                    ║
║        • Fear & Greed + CryptoPanic Sentiment                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import hashlib, hmac, time, logging, os, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from datetime import datetime, date
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger('CIPHER_V3')


# ─────────────────────────────────────────────
#  SENTIMENT
# ─────────────────────────────────────────────
class Sentiment:
    def __init__(self):
        self._cache = {}

    def _get(self, key, ttl=900):
        if key in self._cache:
            v, ts = self._cache[key]
            if time.time() - ts < ttl:
                return v
        return None

    def _set(self, key, val):
        self._cache[key] = (val, time.time())
        return val

    def fear_greed(self):
        c = self._get('fg')
        if c: return c
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5).json()
            v = int(r['data'][0]['value'])
            label = r['data'][0]['value_classification']
            return self._set('fg', {'score': v, 'label': label})
        except:
            return {'score': 50, 'label': 'Neutral'}

    def news(self, coin):
        key = f'cp_{coin}'
        c = self._get(key)
        if c: return c
        try:
            url = f'https://cryptopanic.com/api/free/v1/posts/?auth_token=free&currencies={coin}&filter=hot'
            d = requests.get(url, timeout=5).json().get('results', [])[:10]
            pos = sum(1 for x in d if x.get('votes', {}).get('positive', 0) > x.get('votes', {}).get('negative', 0))
            total = len(d) or 1
            score = pos / total
            return self._set(key, {'score': score, 'pos': pos, 'total': total})
        except:
            return {'score': 0.5, 'pos': 0, 'total': 0}

    def ok(self, side, fg, news_score):
        if side == 'LONG':
            return fg < 80 and news_score >= 0.35
        else:
            return fg > 20 and news_score <= 0.65


# ─────────────────────────────────────────────
#  BINGX API
# ─────────────────────────────────────────────
class BingX:
    BASE = 'https://open-api.bingx.com'

    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.sess = requests.Session()
        self.sess.headers.update({'X-BX-APIKEY': key})

    def _sign(self, p):
        q = '&'.join(f'{k}={v}' for k, v in sorted(p.items()))
        return hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest()

    def _get(self, path, p=None):
        p = p or {}
        p['timestamp'] = int(time.time() * 1000)
        p['signature'] = self._sign(p)
        r = self.sess.get(f'{self.BASE}{path}', params=p, timeout=10)
        d = r.json()
        if d.get('code', 0) != 0:
            raise Exception(f"API {path}: {d.get('msg')}")
        return d.get('data', d)

    def _post(self, path, p=None):
        p = p or {}
        p['timestamp'] = int(time.time() * 1000)
        p['signature'] = self._sign(p)
        r = self.sess.post(f'{self.BASE}{path}', params=p, timeout=10)
        d = r.json()
        if d.get('code', 0) != 0:
            raise Exception(f"API {path}: {d.get('msg')}")
        return d.get('data', d)

    def klines(self, sym, interval, limit=200):
        d = self._get('/openApi/swap/v3/quote/klines',
                      {'symbol': sym, 'interval': interval, 'limit': limit})
        df = pd.DataFrame(d, columns=['time','open','high','low','close','volume','_','_2'])
        df = df[['time','open','high','low','close','volume']].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df.set_index('time')

    def ticker(self, sym):
        d = self._get('/openApi/swap/v2/quote/ticker', {'symbol': sym})
        return d[0] if isinstance(d, list) else d

    def orderbook(self, sym, depth=20):
        return self._get('/openApi/swap/v2/quote/depth', {'symbol': sym, 'limit': depth})

    def open_interest(self, sym):
        try:
            d = self._get('/openApi/swap/v2/quote/openInterest', {'symbol': sym})
            return float(d.get('openInterest', 0))
        except:
            return 0.0

    def funding_rate(self, sym):
        try:
            d = self._get('/openApi/swap/v2/quote/premiumIndex', {'symbol': sym})
            if isinstance(d, list): d = d[0]
            return float(d.get('lastFundingRate', 0))
        except:
            return 0.0

    def taker_ratio(self, sym):
        """Taker Buy/Sell Oranı — alıcı/satıcı dengesi"""
        try:
            d = self._get('/openApi/swap/v2/quote/takerlongshortRatio',
                          {'symbol': sym, 'period': '1h', 'limit': 5})
            if not d: return 0.5
            buy_vol  = float(d[-1].get('buyVol', 0.5))
            sell_vol = float(d[-1].get('sellVol', 0.5))
            ratio = buy_vol / (buy_vol + sell_vol + 1e-9)
            return ratio  # 0.5+ → alıcılar baskın, 0.5- → satıcılar baskın
        except:
            return 0.5

    def set_leverage(self, sym, lev):
        try:
            self._post('/openApi/swap/v2/trade/leverage',
                       {'symbol': sym, 'side': 'LONG', 'leverage': lev})
            self._post('/openApi/swap/v2/trade/leverage',
                       {'symbol': sym, 'side': 'SHORT', 'leverage': lev})
        except Exception as e:
            log.warning(f"Kaldıraç {sym}: {e}")

    def place_order(self, sym, side, qty, sl=None, tp=None, paper=True):
        path = '/openApi/swap/v2/trade/order/test' if paper else '/openApi/swap/v2/trade/order'
        pos  = 'LONG' if side == 'BUY' else 'SHORT'
        r    = self._post(path, {
            'symbol': sym, 'side': side,
            'positionSide': pos, 'type': 'MARKET',
            'quoteOrderQty': qty
        })
        for price, otype in [(sl, 'STOP_MARKET'), (tp, 'TAKE_PROFIT_MARKET')]:
            if price:
                try:
                    cs = 'SELL' if side == 'BUY' else 'BUY'
                    self._post(path, {
                        'symbol': sym, 'side': cs,
                        'positionSide': pos, 'type': otype,
                        'stopPrice': round(price, 4),
                        'closePosition': True
                    })
                except Exception as e:
                    log.warning(f"Koşullu emir {sym}: {e}")
        return r


# ─────────────────────────────────────────────
#  TEKNİK ANALİZ
# ─────────────────────────────────────────────
class TA:

    @staticmethod
    def ema(arr, period):
        k = 2 / (period + 1)
        v = float(arr[0])
        for x in arr[1:]: v = float(x) * k + v * (1 - k)
        return v

    @staticmethod
    def ema_arr(arr, period):
        k = 2 / (period + 1)
        r = [float(arr[0])]
        for x in arr[1:]: r.append(float(x) * k + r[-1] * (1 - k))
        return np.array(r)

    @staticmethod
    def atr(df, period=14):
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def rsi(arr, period=14):
        if len(arr) < period + 1: return 50.0
        d = np.diff(arr[-period-1:])
        g = d[d > 0].sum() / period
        l = -d[d < 0].sum() / period
        return 100.0 if l == 0 else 100 - 100 / (1 + g / (l + 1e-9))

    @staticmethod
    def adx(df, period=14):
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        pdm = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1], 0), 0)
        mdm = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:], 0), 0)
        tr  = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        atr = np.mean(tr[-period:]) + 1e-9
        pdi = 100 * np.mean(pdm[-period:]) / atr
        mdi = 100 * np.mean(mdm[-period:]) / atr
        dx  = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
        return float(dx), float(pdi), float(mdi)

    @staticmethod
    def supertrend(df, period=10, mult=3.0):
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        atr_arr = []
        for i in range(1, len(c)):
            tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            atr_arr.append(tr)
        atr = np.array(atr_arr)
        upper = (h[1:]+l[1:])/2 + mult * atr
        lower = (h[1:]+l[1:])/2 - mult * atr
        trend = np.ones(len(c)-1)
        for i in range(1, len(trend)):
            if c[i] > upper[i-1]:   trend[i] = 1
            elif c[i] < lower[i-1]: trend[i] = -1
            else:                    trend[i] = trend[i-1]
        return int(trend[-1])

    @staticmethod
    def cvd(df):
        c, v, o = df['close'].values, df['volume'].values, df['open'].values
        delta   = np.where(c > o, v, np.where(c < o, -v, 0))
        arr     = np.cumsum(delta)
        recent  = float(np.mean(arr[-5:]))
        prev    = float(np.mean(arr[-10:-5]))
        return recent, recent > 0, recent > prev

    @staticmethod
    def obi(ob):
        try:
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            bv = sum(float(b[1]) for b in bids[:10])
            av = sum(float(a[1]) for a in asks[:10])
            return bv / (av + 1e-9)
        except:
            return 1.0

    @staticmethod
    def volume_spike(df, mult=1.5):
        v = df['volume'].values
        avg = np.mean(v[-20:-1])
        return float(v[-1]) > avg * mult, float(v[-1]) / (avg + 1e-9)

    # ── Yapısal Trend: HH/HL veya LH/LL ──────
    @staticmethod
    def structural_trend(df, lookback=5):
        """
        Gerçek trend tespiti — fiyat yapısına bakır
        HH+HL → BULL, LH+LL → BEAR
        """
        h = df['high'].values
        l = df['low'].values

        # Son 5 swing high ve low bul
        swing_highs = []
        swing_lows  = []
        for i in range(2, len(h)-2):
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                swing_highs.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                swing_lows.append(l[i])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'NEUTRAL'

        sh = swing_highs[-lookback:]
        sl = swing_lows[-lookback:]

        hh = all(sh[i] > sh[i-1] for i in range(1, len(sh)))  # Higher Highs
        hl = all(sl[i] > sl[i-1] for i in range(1, len(sl)))  # Higher Lows
        lh = all(sh[i] < sh[i-1] for i in range(1, len(sh)))  # Lower Highs
        ll = all(sl[i] < sl[i-1] for i in range(1, len(sl)))  # Lower Lows

        if hh and hl: return 'BULL'
        if lh and ll: return 'BEAR'

        # Kısmi kontrol (en az 2 nokta)
        if sh[-1] > sh[-2] and sl[-1] > sl[-2]: return 'BULL'
        if sh[-1] < sh[-2] and sl[-1] < sl[-2]: return 'BEAR'

        return 'NEUTRAL'

    # ── HTF Trend (EMA + Supertrend + Yapısal) ──
    @staticmethod
    def htf_trend(df_daily, df_4h):
        c_d  = df_daily['close'].values
        c_4h = df_4h['close'].values

        ema50_d  = TA.ema(c_d, 50)
        ema200_d = TA.ema(c_d, 200)
        ema20_4h = TA.ema(c_4h, 20)
        ema50_4h = TA.ema(c_4h, 50)

        st_d  = TA.supertrend(df_daily)
        st_4h = TA.supertrend(df_4h)
        adx_val, pdi, mdi = TA.adx(df_4h)

        # Yapısal trend
        struct_d  = TA.structural_trend(df_daily, 3)
        struct_4h = TA.structural_trend(df_4h, 4)

        # Son 3 günlük kapanış yönü
        recent_close_up   = c_d[-1] > c_d[-3]
        recent_close_down = c_d[-1] < c_d[-3]

        bull_score = 0
        bear_score = 0

        if ema50_d > ema200_d: bull_score += 1
        else: bear_score += 1

        if c_d[-1] > ema50_d: bull_score += 1
        else: bear_score += 1

        if c_4h[-1] > ema20_4h and ema20_4h > ema50_4h: bull_score += 1
        elif c_4h[-1] < ema20_4h and ema20_4h < ema50_4h: bear_score += 1

        if st_d == 1:  bull_score += 1
        elif st_d == -1: bear_score += 1

        if st_4h == 1:  bull_score += 1
        elif st_4h == -1: bear_score += 1

        if struct_d == 'BULL':  bull_score += 2
        elif struct_d == 'BEAR': bear_score += 2

        if struct_4h == 'BULL':  bull_score += 1
        elif struct_4h == 'BEAR': bear_score += 1

        if recent_close_up:   bull_score += 1
        if recent_close_down: bear_score += 1

        if adx_val > CONFIG['adx_threshold']:
            if pdi > mdi: bull_score += 1
            else:         bear_score += 1

        log.debug(f"  HTF: bull={bull_score} bear={bear_score} adx={adx_val:.1f} struct_d={struct_d} struct_4h={struct_4h}")

        if bull_score >= 6 and bull_score > bear_score + 2:
            return 'BULL', adx_val
        elif bear_score >= 6 and bear_score > bull_score + 2:
            return 'BEAR', adx_val
        return 'NEUTRAL', adx_val

    # ── Destek/Direnç Bazlı SL ────────────────
    @staticmethod
    def support_resistance_sl(df_1h, df_4h, df_daily, side, price):
        """
        Swing High/Low + Pivot + Order Block kombinasyonu
        ile dinamik SL hesapla
        """
        atr_4h  = TA.atr(df_4h)
        atr_1h  = TA.atr(df_1h)
        cfg     = CONFIG

        # 1. Swing High/Low
        h = df_4h['high'].values
        l = df_4h['low'].values
        lookback = cfg['swing_lookback']
        swing_levels = []
        for i in range(2, min(lookback, len(h))-2):
            idx = -(i+2)
            if h[idx] > h[idx-1] and h[idx] > h[idx+1]:
                swing_levels.append(('high', h[idx]))
            if l[idx] < l[idx-1] and l[idx] < l[idx+1]:
                swing_levels.append(('low', l[idx]))

        # 2. Günlük Pivot
        d_h = df_daily['high'].values[-2]
        d_l = df_daily['low'].values[-2]
        d_c = df_daily['close'].values[-2]
        pivot = (d_h + d_l + d_c) / 3
        s1 = 2 * pivot - d_h
        r1 = 2 * pivot - d_l

        # 3. Order Block (son büyük hareketin başlangıcı)
        c  = df_4h['close'].values
        v  = df_4h['volume'].values
        ob_level = None
        ob_lookback = min(cfg['ob_lookback'], len(c)-1)
        for i in range(1, ob_lookback):
            change = abs(c[-i] - c[-i-1]) / (c[-i-1] + 1e-9)
            if change > 0.01 and v[-i] > np.mean(v[-20:]) * 1.5:
                ob_level = c[-i-1]  # Hareketin başladığı yer
                break

        buffer = atr_4h * cfg['pivot_atr_buffer']

        if side == 'LONG':
            # En yakın destek seviyeleri
            candidates = []

            # Swing lows (fiyatın altında)
            for typ, level in swing_levels:
                if typ == 'low' and level < price - atr_1h:
                    candidates.append(level - buffer)

            # S1 pivot
            if s1 < price - atr_1h:
                candidates.append(s1 - buffer)

            # Order block
            if ob_level and ob_level < price - atr_1h:
                candidates.append(ob_level - buffer)

            if candidates:
                # Fiyata en yakın ama çok yakın olmayan destek
                valid = [c for c in candidates if (price - c) / price > cfg['sl_min_pct']]
                if valid:
                    sl = max(valid)  # En yakın destek
                else:
                    sl = price - atr_4h * 2.0
            else:
                sl = price - atr_4h * 2.0

        else:  # SHORT
            candidates = []

            for typ, level in swing_levels:
                if typ == 'high' and level > price + atr_1h:
                    candidates.append(level + buffer)

            if r1 > price + atr_1h:
                candidates.append(r1 + buffer)

            if ob_level and ob_level > price + atr_1h:
                candidates.append(ob_level + buffer)

            if candidates:
                valid = [c for c in candidates if (c - price) / price > cfg['sl_min_pct']]
                if valid:
                    sl = min(valid)
                else:
                    sl = price + atr_4h * 2.0
            else:
                sl = price + atr_4h * 2.0

        # SL sınır kontrolü
        sl_pct = abs(price - sl) / price
        if sl_pct < cfg['sl_min_pct']:
            sl = price * (1 - cfg['sl_min_pct']) if side == 'LONG' else price * (1 + cfg['sl_min_pct'])
        elif sl_pct > cfg['sl_max_pct']:
            sl = price * (1 - cfg['sl_max_pct']) if side == 'LONG' else price * (1 + cfg['sl_max_pct'])

        return sl

    # ── Kademeli giriş için trend onayı ──────
    @staticmethod
    def trend_confirmed(df_15m, df_1h, side):
        """
        2. kademe giriş için güçlü trend onayı
        15m + 1h Supertrend aynı yönde mi?
        """
        st_15m = TA.supertrend(df_15m)
        st_1h  = TA.supertrend(df_1h)
        exp    = 1 if side == 'LONG' else -1

        rsi_val = TA.rsi(df_1h['close'].values)
        rsi_ok  = (side == 'LONG' and 35 < rsi_val < 70) or \
                  (side == 'SHORT' and 30 < rsi_val < 65)

        return st_15m == exp and st_1h == exp and rsi_ok

    # ── Multi TF Konsensüs ────────────────────
    @staticmethod
    def mtf_consensus(df_15m, df_1h, df_4h, side):
        exp = 1 if side == 'LONG' else -1
        st_15m = TA.supertrend(df_15m)
        st_1h  = TA.supertrend(df_1h)
        st_4h  = TA.supertrend(df_4h)
        score  = sum([st_15m == exp, st_1h == exp, st_4h == exp])
        return score, (st_15m, st_1h, st_4h)

    # ── RSI Bölgesi ──────────────────────────
    @staticmethod
    def rsi_ok(df, side):
        val = TA.rsi(df['close'].values)
        if side == 'LONG':
            return val < 72, val   # Aşırı alımda değilse
        else:
            return val > 28, val   # Aşırı satımda değilse


# ─────────────────────────────────────────────
#  ÇIKIŞ ANALİZİ
# ─────────────────────────────────────────────
class ExitEngine:
    def __init__(self):
        self.oi_hist = {}

    def update_oi(self, sym, val):
        if sym not in self.oi_hist: self.oi_hist[sym] = []
        self.oi_hist[sym].append(val)
        if len(self.oi_hist[sym]) > 10: self.oi_hist[sym].pop(0)

    def oi_dropping(self, sym):
        h = self.oi_hist.get(sym, [])
        if len(h) < 3: return False
        return np.mean(h[-2:]) < np.mean(h[-4:-2]) * (1 + CONFIG['oi_drop_pct']/100)

    def score(self, sym, side, df_1h, df_daily, df_4h, ob, funding, btc_1h, opened_at):
        pts = 0
        reasons = []

        # 1. OI düşüyor
        if self.oi_dropping(sym):
            pts += 1; reasons.append('OI↓')

        # 2. CVD Divergence — en güçlü sinyal
        c = df_1h['close'].values
        _, cvd_pos, cvd_rising = TA.cvd(df_1h)
        if side == 'LONG' and c[-1] > c[-5] and not cvd_rising:
            pts += 2; reasons.append('CVD_DIV')
        elif side == 'SHORT' and c[-1] < c[-5] and cvd_rising:
            pts += 2; reasons.append('CVD_DIV')

        # 3. OBI tersine döndü
        obi = TA.obi(ob)
        if side == 'LONG' and obi < 0.8:
            pts += 1; reasons.append('OBI_REV')
elif 
