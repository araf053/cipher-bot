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
        elif side == 'SHORT' and obi > 1.25:
            pts += 1; reasons.append('OBI_REV')

        # 4. HTF zayıflama
        adx_val, _, _ = TA.adx(df_4h)
        st_4h = TA.supertrend(df_4h)
        st_d  = TA.supertrend(df_daily)
        if adx_val < 18 or st_4h != st_d:
            pts += 1; reasons.append('HTF_WEAK')

        # 5. Funding aşırı
        if funding > 0.05 or funding < -0.03:
            pts += 2; reasons.append('FUND_EXT')

        # 6. BTC çöküşü
        btc_c = btc_1h['close'].values
        if (btc_c[-1] - btc_c[-4]) / btc_c[-4] < -0.025:
            pts += 3; reasons.append('BTC_CRASH')

        # 7. Volatilite patlaması
        if TA.atr(df_1h, 5) > TA.atr(df_1h, 20) * CONFIG['atr_explosion_mult']:
            pts += 1; reasons.append('VOL_EXP')

        # 8. Pozisyon çok eski
        age_h = (datetime.now() - opened_at).total_seconds() / 3600
        if age_h > CONFIG['max_position_age']:
            pts += 1; reasons.append('TOO_OLD')

        # 9. Yapısal trend bozuldu
        struct = TA.structural_trend(df_4h, 3)
        exp_struct = 'BULL' if side == 'LONG' else 'BEAR'
        if struct not in [exp_struct, 'NEUTRAL']:
            pts += 2; reasons.append('STRUCT_BREAK')

        if pts >= 4: return 'FULL_EXIT', reasons
        if pts >= 2: return 'PARTIAL_EXIT', reasons
        if pts >= 1: return 'TIGHTEN_SL', reasons
        return 'HOLD', reasons


# ─────────────────────────────────────────────
#  PAPER TRACKER
# ─────────────────────────────────────────────
class Paper:
    def __init__(self, balance):
        self.balance = balance
        self.positions = {}   # sym → pos dict
        self.trades    = []
        self.start_bal = balance

    def open(self, sym, side, qty, entry, sl, tp1, tp2, stage=1):
        if sym in self.positions and stage == 1:
            log.warning(f"[PAPER] {sym} zaten açık, atlanıyor")
            return False
        if self.balance < qty:
            log.warning(f"[PAPER] Yetersiz bakiye: {self.balance:.2f}")
            return False

        if sym in self.positions and stage == 2:
            # 2. kademe — mevcut pozisyona ekleme
            pos = self.positions[sym]
            pos['qty']    += qty
            pos['stage']   = 2
            self.balance  -= qty
            log.info(f"[PAPER] STAGE2 {side} {sym} +{qty} USDT │ Toplam:{pos['qty']} USDT")
            return True

        self.positions[sym] = {
            'side': side, 'entry': entry, 'qty': qty,
            'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'tp1_hit': False, 'partial': False,
            'opened': datetime.now(), 'stage': 1
        }
        self.balance -= qty
        log.info(f"[PAPER] OPEN {side} {sym} @{entry:.4f} │ {qty} USDT │ SL:{sl:.4f}")
        return True

    def partial_close(self, sym, price, reason):
        if sym not in self.positions: return 0
        pos  = self.positions[sym]
        half = pos['qty'] / 2
        pnl  = half * (price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
        self.balance  += half + pnl
        pos['qty']    -= half
        pos['partial'] = True
        pos['sl']      = pos['entry']  # SL başa çek
        self.trades.append({'sym': sym, 'pnl': pnl, 'reason': f'PARTIAL'})
        log.info(f"[PAPER] PARTIAL {sym} @{price:.4f} │ PNL:{pnl:+.2f}")
        return pnl

    def close(self, sym, price, reason):
        if sym not in self.positions: return 0, 0, 0
        pos  = self.positions.pop(sym)
        pnl  = pos['qty'] * (price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
        self.balance += pos['qty'] + pnl
        self.trades.append({'sym': sym, 'pnl': pnl, 'reason': reason})
        em = '✅' if pnl > 0 else '❌'
        log.info(f"[PAPER] CLOSE {em} {sym} @{price:.4f} │ {reason} │ PNL:{pnl:+.2f}")
        return pnl, pos['entry'], price

    def check_tp_sl(self, sym, price):
        if sym not in self.positions: return None, 0, 0, 0
        pos  = self.positions[sym]
        side = pos['side']
        hit_sl  = (side == 'LONG'  and price <= pos['sl']) or \
                  (side == 'SHORT' and price >= pos['sl'])
        hit_tp2 = (side == 'LONG'  and price >= pos['tp2']) or \
                  (side == 'SHORT' and price <= pos['tp2'])
        hit_tp1 = not pos['tp1_hit'] and (
            (side == 'LONG'  and price >= pos['tp1']) or
            (side == 'SHORT' and price <= pos['tp1'])
        )

        if hit_sl:
            # SL fiyatını kullan, anlık fiyatı değil
            sl_price = pos['sl']
            pnl, ep, cp = self.close(sym, sl_price, 'SL')
            return 'SL', pnl, ep, cp
        if hit_tp2:
            pnl, ep, cp = self.close(sym, price, 'TP2')
            return 'TP2', pnl, ep, cp
        if hit_tp1:
            pos['tp1_hit'] = True
            pos['sl'] = pos['entry']  # BE'ye çek
            log.info(f"[PAPER] TP1 ✓ {sym} → SL başa çekildi")
            return 'TP1', 0, 0, 0
        return None, 0, 0, 0

    def stats(self):
        pnls = [t['pnl'] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        return {
            'count': len(pnls), 'winning': len(wins),
            'losing': len(pnls) - len(wins),
            'total_pnl': sum(pnls),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            'balance': self.balance
        }


# ─────────────────────────────────────────────
#  BİLDİRİM — GELİŞMİŞ NTFY
# ─────────────────────────────────────────────
class Notifier:
    CMC = {
        'BTC':'bitcoin','ETH':'ethereum','SOL':'solana','BNB':'bnb',
        'AVAX':'avalanche-2','LINK':'chainlink','ARB':'arbitrum',
        'OP':'optimism','DOGE':'dogecoin','APT':'aptos','SUI':'sui',
        'DOT':'polkadot','MATIC':'matic-network','ATOM':'cosmos',
        'INJ':'injective-protocol'
    }

    def __init__(self, ch):
        self.ch  = ch
        self.ok  = bool(ch)
        self.bal_history = []   # Bakiye geçmişi → günlük grafik için

    def _cmc(self, sym):
        coin = sym.replace('-USDT','')
        slug = self.CMC.get(coin, coin.lower())
        return f"https://coinmarketcap.com/currencies/{slug}/"

    def _bingx(self, sym):
        return f"https://bingx.com/en/futures/{sym.replace('-','_')}/"

    # ── Header encode yardımcısı ─────────────
    @staticmethod
    def _h(text):
        """Header için emoji ve unicode karakterleri ASCII'ye çevir"""
        return text.encode('ascii', 'ignore').decode('ascii').strip()

    # ── Düz metin bildirimi ───────────────────
    def _send(self, title, body, sym=None, priority='default', tags='chart_with_upwards_trend'):
        if not self.ok:
            log.info(f"[NOTIF] {title}")
            return
        try:
            h = {
                'Title'   : self._h(title),
                'Priority': priority,
                'Tags'    : tags
            }
            if sym:
                h['Click']   = self._cmc(sym)
                h['Actions'] = (
                    f"view, CoinMarketCap, {self._cmc(sym)}; "
                    f"view, BingX, {self._bingx(sym)}"
                )
            requests.post(
                f'https://ntfy.sh/{self.ch}',
                data=body.encode('utf-8'),
                headers=h, timeout=5
            )
        except Exception as e:
            log.warning(f"Bildirim hatası: {e}")

    # ── PNG grafik bildirimi ──────────────────
    def _send_image(self, title, img_bytes, sym=None, priority='default'):
        if not self.ok: return
        try:
            h = {
                'Title'   : self._h(title),
                'Priority': priority,
                'Filename': 'chart.png',
                'Tags'    : 'bar_chart'
            }
            if sym:
                h['Click']   = self._cmc(sym)
                h['Actions'] = (
                    f"view, CoinMarketCap, {self._cmc(sym)}; "
                    f"view, BingX, {self._bingx(sym)}"
                )
            requests.post(
                f'https://ntfy.sh/{self.ch}',
                data=img_bytes,
                headers=h, timeout=10
            )
        except Exception as e:
            log.warning(f"Görsel bildirim hatası: {e}")

    # ── Giriş grafiği ─────────────────────────
    def _entry_chart(self, sym, side, price, sl, tp1, tp2, passed, failed, score):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                     gridspec_kw={'width_ratios': [1, 1.6]})
            fig.patch.set_facecolor('#0d1117')

            # ── Sol: Fiyat Seviyeleri ──
            ax = axes[0]
            ax.set_facecolor('#0d1117')
            ax.set_xlim(0, 1)

            is_long  = side == 'LONG'
            clr_main = '#00c853' if is_long else '#ff1744'
            clr_tp   = '#00e676'
            clr_sl   = '#ff1744'
            clr_tp2  = '#69f0ae'

            levels = [
                (tp2,   f'TP2  {tp2:.4f}',   clr_tp2,  0.95),
                (tp1,   f'TP1  {tp1:.4f}',   clr_tp,   0.75),
                (price, f'GİRİŞ {price:.4f}', '#ffffff', 0.50),
                (sl,    f'SL   {sl:.4f}',    clr_sl,   0.25),
            ] if is_long else [
                (sl,    f'SL   {sl:.4f}',    clr_sl,   0.95),
                (price, f'GİRİŞ {price:.4f}', '#ffffff', 0.75),
                (tp1,   f'TP1  {tp1:.4f}',   clr_tp,   0.50),
                (tp2,   f'TP2  {tp2:.4f}',   clr_tp2,  0.25),
            ]

            for _, label, color, y in levels:
                ax.axhline(y=y, color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                ax.text(0.05, y + 0.02, label, color=color,
                        fontsize=9, fontweight='bold', va='bottom',
                        fontfamily='monospace')

            risk  = abs(price - sl)
            rr    = abs(tp2 - price) / (risk + 1e-9)
            sl_pct = risk / price * 100
            mode_txt = 'LONG' if is_long else 'SHORT'

            ax.text(0.5, 0.01,
                    f"{mode_txt}  |  R/R 1:{rr:.1f}  |  SL %{sl_pct:.1f}",
                    color=clr_main, fontsize=8, ha='center', va='bottom',
                    transform=ax.transAxes, fontweight='bold')

            ax.set_title(f'{sym}', color='white', fontsize=12, fontweight='bold', pad=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

            # ── Sağ: Sinyal Skoru ──
            ax2 = axes[1]
            ax2.set_facecolor('#0d1117')
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.set_xticks([]); ax2.set_yticks([])
            for spine in ax2.spines.values(): spine.set_edgecolor('#30363d')

            ax2.set_title(f'Sinyal Skoru: {score}', color='#f0f6fc',
                          fontsize=11, fontweight='bold', pad=8)

            # Geçen sinyaller
            y_pos = 0.93
            ax2.text(0.05, y_pos, '[GEC] SINYALLER', color='#00c853',
                     fontsize=8, fontweight='bold', transform=ax2.transAxes)
            y_pos -= 0.07
            for p in passed[:8]:
                ax2.text(0.05, y_pos, f'  + {p}', color='#8b949e',
                         fontsize=7.5, transform=ax2.transAxes, fontfamily='monospace')
                y_pos -= 0.065

            # Kalan sinyaller
            if failed:
                y_pos -= 0.02
                ax2.text(0.05, y_pos, '[KAL] SINYALLER', color='#ff1744',
                         fontsize=8, fontweight='bold', transform=ax2.transAxes)
                y_pos -= 0.07
                for f in failed[:5]:
                    ax2.text(0.05, y_pos, f'  - {f}', color='#6e7681',
                             fontsize=7.5, transform=ax2.transAxes, fontfamily='monospace')
                    y_pos -= 0.065

            # Skor barı
            total_possible = score + len(failed)
            bar_w = score / max(total_possible, 1)
            ax2.add_patch(mpatches.FancyBboxPatch(
                (0.05, 0.04), 0.9, 0.06,
                boxstyle='round,pad=0.01',
                facecolor='#21262d', edgecolor='#30363d'
            ))
            ax2.add_patch(mpatches.FancyBboxPatch(
                (0.05, 0.04), 0.9 * bar_w, 0.06,
                boxstyle='round,pad=0.01',
                facecolor=clr_main, edgecolor='none', alpha=0.8
            ))
            ax2.text(0.5, 0.07, f'{score} / {total_possible}',
                     color='white', fontsize=8, ha='center', va='center',
                     transform=ax2.transAxes, fontweight='bold')

            plt.tight_layout(pad=1.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                        facecolor='#0d1117')
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            log.warning(f"Giriş grafiği hatası: {e}")
            return None

    # ── Günlük rapor grafiği ──────────────────
    def _daily_chart(self, trades, start_bal, current_bal):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            fig.patch.set_facecolor('#0d1117')

            # ── Sol: Bakiye eğrisi ──
            ax1 = axes[0]
            ax1.set_facecolor('#161b22')

            bal = start_bal
            bal_curve = [bal]
            for t in trades:
                bal += t['pnl']
                bal_curve.append(bal)

            xs = list(range(len(bal_curve)))
            colors_line = ['#00c853' if b >= start_bal else '#ff1744'
                           for b in bal_curve]

            for i in range(len(xs)-1):
                clr = '#00c853' if bal_curve[i+1] >= bal_curve[i] else '#ff1744'
                ax1.plot(xs[i:i+2], bal_curve[i:i+2], color=clr, linewidth=2)

            ax1.fill_between(xs, bal_curve, start_bal,
                             where=[b >= start_bal for b in bal_curve],
                             alpha=0.15, color='#00c853')
            ax1.fill_between(xs, bal_curve, start_bal,
                             where=[b < start_bal for b in bal_curve],
                             alpha=0.15, color='#ff1744')

            ax1.axhline(y=start_bal, color='#8b949e', linestyle='--',
                        linewidth=1, alpha=0.5)
            ax1.set_title('Bakiye Grafiği', color='white', fontsize=10,
                          fontweight='bold')
            ax1.set_facecolor('#161b22')
            ax1.tick_params(colors='#8b949e', labelsize=7)
            ax1.yaxis.label.set_color('#8b949e')
            for spine in ax1.spines.values(): spine.set_edgecolor('#30363d')
            ax1.set_xlabel('İşlem #', color='#8b949e', fontsize=8)
            ax1.set_ylabel('USDT', color='#8b949e', fontsize=8)

            # ── Sağ: İşlem dağılımı ──
            ax2 = axes[1]
            ax2.set_facecolor('#161b22')

            pnls   = [t['pnl'] for t in trades]
            wins   = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            if pnls:
                # Bar chart — her işlem
                bar_colors = ['#00c853' if p > 0 else '#ff1744' for p in pnls]
                xs2 = list(range(len(pnls)))
                bars = ax2.bar(xs2, pnls, color=bar_colors, alpha=0.8, width=0.6)
                ax2.axhline(y=0, color='#8b949e', linewidth=0.8, alpha=0.5)

                # İstatistik kutusu
                wr   = len(wins) / len(pnls) * 100
                total = sum(pnls)
                avg_w = sum(wins) / len(wins) if wins else 0
                avg_l = sum(losses) / len(losses) if losses else 0

                stats_txt = (
                    f"İşlem: {len(pnls)}   Win: %{wr:.0f}\n"
                    f"Ort Kazanç: +{avg_w:.2f}$\n"
                    f"Ort Kayıp : {avg_l:.2f}$\n"
                    f"Toplam PNL: {total:+.2f}$"
                )
                ax2.text(0.98, 0.98, stats_txt,
                         transform=ax2.transAxes,
                         color='#f0f6fc', fontsize=7.5,
                         va='top', ha='right',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='#21262d',
                                   edgecolor='#30363d', alpha=0.9))

            ax2.set_title('İşlem Dağılımı', color='white', fontsize=10,
                          fontweight='bold')
            ax2.tick_params(colors='#8b949e', labelsize=7)
            for spine in ax2.spines.values(): spine.set_edgecolor('#30363d')
            ax2.set_xlabel('İşlem #', color='#8b949e', fontsize=8)
            ax2.set_ylabel('PNL (USDT)', color='#8b949e', fontsize=8)

            plt.tight_layout(pad=1.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                        facecolor='#0d1117')
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            log.warning(f"Rapor grafiği hatası: {e}")
            return None

    # ── GİRİŞ BİLDİRİMİ ─────────────────────
    def entry(self, sym, side, price, sl, tp1, tp2, qty, stage, paper, passed=None, failed=None, score=0):
        mode   = 'PAPER' if paper else 'CANLI'
        em     = '🟢 LONG' if side == 'LONG' else '🔴 SHORT'
        kd     = f'KADEME {stage}'
        sl_pct = abs(price - sl) / price * 100
        rr     = abs(tp2 - price) / (abs(price - sl) + 1e-9)
        title  = f"{em} — {sym} [{kd}] [{mode}]"

        body = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 {sym}  |  {em}  |  {kd}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Giriş : {price:.4f}\n"
            f"🎯 TP1   : {tp1:.4f}\n"
            f"🎯 TP2   : {tp2:.4f}\n"
            f"🛑 SL    : {sl:.4f}  (%{sl_pct:.1f})\n"
            f"⚖️  R/R   : 1:{rr:.1f}\n"
            f"💰 Miktar: {qty} USDT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        # Grafik üret ve gönder
        if passed is not None:
            img = self._entry_chart(sym, side, price, sl, tp1, tp2,
                                    passed, failed or [], score)
            if img:
                self._send_image(title, img, sym, priority='high')
                return

        self._send(title, body, sym, priority='high')

    # ── ÇIKIŞ BİLDİRİMİ ─────────────────────
    def exit_notif(self, sym, reason, pnl, balance, entry_p, close_p, paper):
        mode   = 'PAPER' if paper else 'CANLI'
        em     = '✅ KAZANÇ' if pnl > 0 else '❌ KAYIP'
        chg    = (close_p - entry_p) / entry_p * 100
        title  = f"{em} — {sym}  {pnl:+.2f} USDT [{mode}]"
        body   = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 {sym}  |  {em}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔓 Giriş : {entry_p:.4f}\n"
            f"🔒 Çıkış : {close_p:.4f}  ({chg:+.2f}%)\n"
            f"📊 Sebep : {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 PNL   : {pnl:+.2f} USDT\n"
            f"🏦 Bakiye: {balance:.2f} USDT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        priority = 'high' if pnl > 0 else 'default'
        tags     = 'white_check_mark' if pnl > 0 else 'x'
        self._send(title, body, sym, priority=priority, tags=tags)

    # ── KISMİ ÇIKIŞ ──────────────────────────
    def partial_notif(self, sym, reasons, pnl):
        body = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ {sym} — KISMİ ÇIKIŞ\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Sebep : {', '.join(reasons)}\n"
            f"💰 PNL   : {pnl:+.2f} USDT\n"
            f"🔒 SL başa çekildi\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send(f"⚠️ KISMİ ÇIKIŞ — {sym}", body, sym, tags='warning')

    # ── 2. KADEME ────────────────────────────
    def stage2_notif(self, sym, side, price, qty, paper):
        mode = 'PAPER' if paper else 'CANLI'
        em   = '🟢' if side == 'LONG' else '🔴'
        body = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"➕ {sym} | 2. KADEME GİRİŞ\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{em} {side} @ {price:.4f}\n"
            f"💰 Eklenen : {qty} USDT\n"
            f"📊 Trend onaylandı ✅\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send(f"➕ 2. KADEME — {sym} [{mode}]", body, sym, priority='high')

    # ── GÜNLÜK RAPOR ─────────────────────────
    def daily_report(self, stats, paper, trades=None, start_bal=None):
        mode  = 'PAPER' if paper else 'CANLI'
        pnl   = stats['total_pnl']
        em    = '📈' if pnl >= 0 else '📉'
        wr    = stats['win_rate']
        title = f"{em} GÜNLÜK RAPOR [{mode}]"

        body = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{em} GÜNLÜK RAPOR [{mode}]\n"
            f"📅 {datetime.now().strftime('%d %B %Y')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 İşlem  : {stats['count']}\n"
            f"✅ Kazanç : {stats['winning']}\n"
            f"❌ Kayıp  : {stats['losing']}\n"
            f"🎯 Win    : %{wr:.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 PNL    : {pnl:+.2f} USDT\n"
            f"🏦 Bakiye : {stats['balance']:.2f} USDT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        # Grafik üret
        if trades and start_bal:
            img = self._daily_chart(trades, start_bal, stats['balance'])
            if img:
                self._send_image(title, img, priority='default')
                return

        self._send(title, body, tags='bar_chart')


# ─────────────────────────────────────────────
#  ANA BOT — CIPHER V3
# ─────────────────────────────────────────────
class CipherV3:
    def __init__(self):
        self.cfg    = CONFIG
        self.api    = BingX(self.cfg['api_key'], self.cfg['api_secret'])
        self.paper  = Paper(self.cfg['initial_balance'])
        self.exit_e = ExitEngine()
        self.sent   = Sentiment()
        self.notif  = Notifier(self.cfg['ntfy_channel'])

        self.is_paper      = self.cfg['paper_trading']
        self.reentry_times = {}   # sym → son kapanma zamanı
        self.oi_prev       = {}
        self.btc_cache     = {}
        self.last_reset    = date.today()

        # 2. kademe bekleyenler: sym → signal bilgisi
        self.pending_stage2 = {}

        log.info("═" * 55)
        log.info("  CIPHER V3 BAŞLADI")
        log.info(f"  Mod     : {'📝 PAPER' if self.is_paper else '💰 CANLI'}")
        log.info(f"  Coinler : {len(self.cfg['symbols'])} adet")
        log.info(f"  Kaldıraç: {self.cfg['leverage']}x")
        log.info("═" * 55)

        self.notif._send(
            "🚀 CIPHER V3 BAŞLADI",
            f"{'📝 Paper' if self.is_paper else '💰 CANLI'}\n"
            f"Coinler: {len(self.cfg['symbols'])} majör\n"
            f"Kaldıraç: {self.cfg['leverage']}x\n"
            f"Bakiye: {self.cfg['initial_balance']} USDT",
            tags='rocket'
        )

    # ── BTC cache ─────────────────────────────
    def _btc(self):
        now = time.time()
        if 'ts' in self.btc_cache and now - self.btc_cache['ts'] < 300:
            return self.btc_cache['d']
        d = {
            '1h'   : self.api.klines('BTC-USDT', '1h',  200),
            '4h'   : self.api.klines('BTC-USDT', '4h',  200),
            'daily': self.api.klines('BTC-USDT', '1d',  200)
        }
        self.btc_cache = {'ts': now, 'd': d}
        return d

    # ── Günlük reset ──────────────────────────
    def _daily_reset(self):
        today = date.today()
        if today > self.last_reset:
            stats = self.paper.stats()
            self.notif.daily_report(stats, self.is_paper,
                                    trades=self.paper.trades,
                                    start_bal=self.paper.start_bal)
            self.last_reset = today
            log.info("🔄 Günlük reset")

    # ── Reentry engeli ────────────────────────
    def _can_enter(self, sym):
        last = self.reentry_times.get(sym, 0)
        wait = self.cfg['reentry_wait_secs']
        if time.time() - last < wait:
            mins = int((wait - (time.time() - last)) / 60)
            log.info(f"  ⏳ {sym} → {mins}dk bekleniyor")
            return False
        return True

    # ── Sinyal analizi ────────────────────────
    def _analyze(self, sym, btc):
        try:
            df_15m   = self.api.klines(sym, '15m', 100)
            df_1h    = self.api.klines(sym, '1h',  200)
            df_4h    = self.api.klines(sym, '4h',  200)
            df_daily = self.api.klines(sym, '1d',  200)
            ob       = self.api.orderbook(sym, 20)
            oi       = self.api.open_interest(sym)
            funding  = self.api.funding_rate(sym)
            taker    = self.api.taker_ratio(sym)
        except Exception as e:
            log.warning(f"Veri hatası {sym}: {e}")
            return None

        score  = 0
        passed = []
        failed = []
        c      = df_1h['close'].values
        price  = float(c[-1])

        # ── 1. HTF Trend (EMA + Supertrend + Yapısal) ──
        trend, adx_val = TA.htf_trend(df_daily, df_4h)
        if trend == 'BULL':
            side = 'LONG'; score += 3; passed.append(f'HTF_BULL(ADX:{adx_val:.0f})')
        elif trend == 'BEAR':
            side = 'SHORT'; score += 3; passed.append(f'HTF_BEAR(ADX:{adx_val:.0f})')
        else:
            failed.append('HTF_NEUTRAL'); return None

        # ── 2. Multi TF Konsensüs ──
        mtf_score, sts = TA.mtf_consensus(df_15m, df_1h, df_4h, side)
        if mtf_score >= 2:
            score += mtf_score; passed.append(f'MTF({mtf_score}/3)')
        elif mtf_score == 1:
            passed.append(f'MTF_WEAK({mtf_score}/3)')
        else:
            failed.append(f'MTF_FAIL({sts[0]}/{sts[1]}/{sts[2]})'); return None

        # ── 3. Yapısal Trend (HH/HL) ──
        struct_4h = TA.structural_trend(df_4h, 4)
        exp_s = 'BULL' if side == 'LONG' else 'BEAR'
        if struct_4h == exp_s:
            score += 2; passed.append(f'STRUCT_OK({struct_4h})')
        elif struct_4h == 'NEUTRAL':
            passed.append('STRUCT_NEUTRAL')
        else:
            failed.append(f'STRUCT_AGAINST({struct_4h})'); return None

        # ── 4. RSI ──
        rsi_good, rsi_val = TA.rsi_ok(df_1h, side)
        if rsi_good:
            score += 1; passed.append(f'RSI_OK({rsi_val:.0f})')
        else:
            failed.append(f'RSI_BAD({rsi_val:.0f})'); return None

        # ── 5. CVD ──
        cvd_val, cvd_pos, cvd_rising = TA.cvd(df_1h)
        cvd_ok = (side == 'LONG' and cvd_pos and cvd_rising) or \
                 (side == 'SHORT' and not cvd_pos and not cvd_rising)
        if cvd_ok:
            score += 1; passed.append(f'CVD_OK({cvd_val:.0f})')
        else:
            failed.append(f'CVD_FAIL')

        # ── 6. OBI ──
        obi_val = TA.obi(ob)
        if side == 'LONG' and obi_val >= self.cfg['obi_threshold']:
            score += 1; passed.append(f'OBI_BULL({obi_val:.2f})')
        elif side == 'SHORT' and obi_val <= (1 / self.cfg['obi_threshold']):
            score += 1; passed.append(f'OBI_BEAR({obi_val:.2f})')
        else:
            failed.append(f'OBI_NEUTRAL({obi_val:.2f})')

        # ── 7. Taker Buy/Sell Oranı ──
        if side == 'LONG' and taker > 0.52:
            score += 1; passed.append(f'TAKER_BUY({taker:.2f})')
        elif side == 'SHORT' and taker < 0.48:
            score += 1; passed.append(f'TAKER_SELL({taker:.2f})')
        else:
            failed.append(f'TAKER_NEUTRAL({taker:.2f})')

        # ── 8. Volume Spike ──
        vol_ok, vol_ratio = TA.volume_spike(df_1h, self.cfg['volume_spike_mult'])
        if vol_ok:
            score += 1; passed.append(f'VOL({vol_ratio:.1f}x)')
        else:
            failed.append(f'VOL_LOW({vol_ratio:.1f}x)')

        # ── 9. OI ──
        prev_oi = self.oi_prev.get(sym, oi)
        oi_chg  = (oi - prev_oi) / (prev_oi + 1e-9) * 100
        self.oi_prev[sym] = oi
        self.exit_e.update_oi(sym, oi)
        if oi_chg >= self.cfg['oi_change_pct']:
            score += 1; passed.append(f'OI_UP({oi_chg:+.1f}%)')
        else:
            failed.append(f'OI_FLAT({oi_chg:+.1f}%)')

        # ── 10. Funding Rate ──
        f_ok = self.cfg['funding_min'] <= funding <= self.cfg['funding_max']
        if f_ok:
            score += 1; passed.append(f'FUND_OK({funding:.4f})')
        else:
            failed.append(f'FUND_BAD({funding:.4f})'); return None

        # ── 11. BTC Korelasyon ──
        btc_c   = btc['1h']['close'].values
        btc_chg = (btc_c[-1] - btc_c[-4]) / btc_c[-4]
        if side == 'LONG' and btc_chg > self.cfg['btc_drop_limit']:
            score += 1; passed.append(f'BTC_OK({btc_chg:+.2%})')
        elif side == 'SHORT':
            score += 1; passed.append('BTC_SHORT_OK')
        else:
            failed.append(f'BTC_DROP({btc_chg:+.2%})'); return None

        # ── 12. Sentiment ──
        try:
            coin = sym.replace('-USDT', '')
            fg   = self.sent.fear_greed()
            news = self.sent.news(coin)
            if self.sent.ok(side, fg['score'], news['score']):
                score += 1; passed.append(f"SENT_OK(FG:{fg['score']})")
            else:
                failed.append(f"SENT_FAIL(FG:{fg['score']})")
                return None
        except:
            passed.append('SENT_SKIP')

        # ── Minimum skor ──
        if score < 8:
            log.debug(f"  {sym} skor düşük: {score} │ ❌ {', '.join(failed[:3])}")
            return None

        # ── Destek/Direnç Bazlı SL ──
        sl = TA.support_resistance_sl(df_1h, df_4h, df_daily, side, price)

        # ── TP Hesabı (Risk/Reward bazlı) ──
        risk = abs(price - sl)
        if side == 'LONG':
            tp1 = price + risk * self.cfg['tp1_rr']
            tp2 = price + risk * self.cfg['tp2_rr']
        else:
            tp1 = price - risk * self.cfg['tp1_rr']
            tp2 = price - risk * self.cfg['tp2_rr']

        return {
            'sym': sym, 'side': side, 'price': price,
            'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'score': score, 'passed': passed, 'failed': failed,
            'funding': funding,
            'df_1h': df_1h, 'df_4h': df_4h, 'df_daily': df_daily,
            'df_15m': df_15m, 'ob': ob
        }

    # ── 2. Kademe kontrolü ────────────────────
    def _check_stage2(self, btc):
        """Bekleyen 2. kademe girişleri kontrol et"""
        for sym, sig in list(self.pending_stage2.items()):
            if sym not in self.paper.positions:
                del self.pending_stage2[sym]
                continue
            try:
                df_15m = self.api.klines(sym, '15m', 100)
                df_1h  = self.api.klines(sym, '1h',  100)
                if TA.trend_confirmed(df_15m, df_1h, sig['side']):
                    qty   = self.cfg['entry2_usdt']
                    price = float(self.api.ticker(sym).get('lastPrice', sig['price']))
                    ok    = self.paper.open(sym, sig['side'], qty, price,
                                            sig['sl'], sig['tp1'], sig['tp2'], stage=2)
                    if ok:
                        self.notif.stage2_notif(sym, sig['side'], price, qty, self.is_paper)
                        log.info(f"  ➕ STAGE2 {sym} │ {qty} USDT @ {price:.4f}")
                    del self.pending_stage2[sym]
            except Exception as e:
                log.warning(f"Stage2 {sym}: {e}")

    # ── Pozisyon izleme ───────────────────────
    def _monitor(self, btc):
        for sym in list(self.paper.positions.keys()):
            try:
                price = float(self.api.ticker(sym).get('lastPrice', 0))
                pos   = self.paper.positions.get(sym)
                if not pos: continue

                # TP/SL
                reason, pnl, ep, cp = self.paper.check_tp_sl(sym, price)
                if reason and reason != 'TP1':
                    self.notif.exit_notif(sym, reason, pnl, self.paper.balance,
                                          ep, cp, self.is_paper)
                    self.reentry_times[sym] = time.time()
                    if sym in self.pending_stage2:
                        del self.pending_stage2[sym]
                    continue

                # Çıkış analizi
                df_1h    = self.api.klines(sym, '1h',  100)
                df_4h    = self.api.klines(sym, '4h',  100)
                df_daily = self.api.klines(sym, '1d',  100)
                ob       = self.api.orderbook(sym, 20)
                funding  = self.api.funding_rate(sym)

                sig, reasons = self.exit_e.score(
                    sym, pos['side'], df_1h, df_daily, df_4h,
                    ob, funding, btc['1h'], pos['opened']
                )

                if sig == 'FULL_EXIT':
                    pnl, ep, cp = self.paper.close(sym, price, f"EXIT")
                    self.notif.exit_notif(sym, ', '.join(reasons), pnl,
                                          self.paper.balance, ep, cp, self.is_paper)
                    self.reentry_times[sym] = time.time()
                    log.info(f"  🚪 FULL EXIT {sym} │ {', '.join(reasons)}")

                elif sig == 'PARTIAL_EXIT' and not pos.get('partial'):
                    pnl = self.paper.partial_close(sym, price, ','.join(reasons))
                    self.notif.partial_notif(sym, reasons, pnl)

            except Exception as e:
                log.warning(f"İzleme {sym}: {e}")

    # ── Ana Tarama ────────────────────────────
    def scan(self):
        self._daily_reset()

        log.info(f"\n{'═'*55}")
        log.info(f"🔍 TARAMA │ {datetime.now().strftime('%H:%M:%S')} │ "
                 f"Açık:{len(self.paper.positions)} │ "
                 f"Bakiye:{self.paper.balance:.2f}")

        try:
            btc = self._btc()
        except Exception as e:
            log.error(f"BTC verisi: {e}"); return

        # Pozisyon izle
        self._monitor(btc)

        # 2. kademe kontrol
        self._check_stage2(btc)

        # Max pozisyon
        if len(self.paper.positions) >= self.cfg['max_positions']:
            log.info(f"Max pozisyon dolu ({self.cfg['max_positions']})"); return

        # Coin tarama
        for sym in self.cfg['symbols']:
            if sym in self.paper.positions: continue
            if len(self.paper.positions) >= self.cfg['max_positions']: break
            if not self._can_enter(sym): continue

            log.info(f"  ▷ {sym}...")
            result = self._analyze(sym, btc)

            if result:
                qty   = self.cfg['entry1_usdt']
                price = result['price']
                sl    = result['sl']
                tp1   = result['tp1']
                tp2   = result['tp2']
                side  = result['side']
                sl_pct = abs(price - sl) / price * 100

                log.info(
                    f"  ✅ {sym} {side} │ Skor:{result['score']} │ "
                    f"SL:%{sl_pct:.1f} │ {qty}USDT\n"
                    f"     {', '.join(result['passed'])}"
                )

                if self.is_paper:
                    ok = self.paper.open(sym, side, qty, price, sl, tp1, tp2, stage=1)
                    if ok:
                        self.notif.entry(sym, side, price, sl, tp1, tp2, qty, 1, True,
                                        passed=result['passed'], failed=result['failed'],
                                        score=result['score'])
                        # 2. kademe için kaydet
                        self.pending_stage2[sym] = result
                else:
                    self.api.set_leverage(sym, self.cfg['leverage'])
                    self.api.place_order(
                        sym, 'BUY' if side == 'LONG' else 'SELL',
                        qty, sl=sl, tp=tp1, paper=False
                    )
                    self.notif.entry(sym, side, price, sl, tp1, tp2, qty, 1, False)
                    self.pending_stage2[sym] = result
            else:
                log.info(f"  ✕ {sym} → pas")

            time.sleep(1)

        # Özet
        s = self.paper.stats()
        log.info(
            f"📊 Açık:{len(self.paper.positions)} │ "
            f"WR:%{s['win_rate']:.0f} │ "
            f"PNL:{s['total_pnl']:+.2f}"
        )

    # ── Döngü ────────────────────────────────
    def run(self):
        interval = self.cfg['scan_interval_minutes'] * 60
        while True:
            try:
                self.scan()
            except Exception as e:
                log.error(f"Tarama hatası: {e}")
                self.notif._send("⚠️ HATA", str(e), tags='warning')
            log.info(f"💤 {self.cfg['scan_interval_minutes']}dk...\n")
            time.sleep(interval)


# ─────────────────────────────────────────────
if __name__ == '__main__':
    bot = CipherV3()
    try:
        bot.run()
    except KeyboardInterrupt:
        s = bot.paper.stats()
        bot.notif._send("⏹ DURDURULDU",
                        f"PNL:{s['total_pnl']:+.2f} USDT\nBakiye:{s['balance']:.2f}")
        log.info("Bot durduruldu")
