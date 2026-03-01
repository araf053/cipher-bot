"""
╔══════════════════════════════════════════════════════════════╗
║           CIPHER V2 — Gelişmiş Futures Trading Bot           ║
║                                                              ║
║  GİRİŞ  : HTF Trend + Breakout + Volume + OI + Funding      ║
║           + OBI + CVD + Liquidation + Whale + BTC Korelasyon ║
║  ÇIKIŞ  : OI düşüş + CVD div + OBI dönüş + HTF zayıflama   ║
║  RİSK   : Dinamik boyut + Günlük limit + Tekrar giriş engeli ║
║  KOİNLER: BTC + ETH + SOL + BNB                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import hashlib, hmac, time, logging, os, requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from config import CONFIG

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger('CIPHER_V2')


# ─────────────────────────────────────────────
#  BINGX API CLIENT
# ─────────────────────────────────────────────
class BingXClient:
    BASE = 'https://open-api.bingx.com'

    def __init__(self, key, secret, paper=True):
        self.key    = key
        self.secret = secret
        self.paper  = paper
        self.sess   = requests.Session()
        self.sess.headers.update({'X-BX-APIKEY': key})
        log.info(f"BingX │ {'📝 PAPER' if paper else '💰 GERÇEK'}")

    def _sign(self, p):
        q = '&'.join(f'{k}={v}' for k, v in sorted(p.items()))
        return hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest()

    def _get(self, path, p=None):
        p = p or {}
        p['timestamp'] = int(time.time() * 1000)
        p['signature'] = self._sign(p)
        r = self.sess.get(f'{self.BASE}{path}', params=p, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get('code', 0) != 0:
            raise Exception(f"API: {d.get('msg')}")
        return d.get('data', d)

    def _post(self, path, p=None):
        p = p or {}
        p['timestamp'] = int(time.time() * 1000)
        p['signature'] = self._sign(p)
        r = self.sess.post(f'{self.BASE}{path}', params=p, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get('code', 0) != 0:
            raise Exception(f"API: {d.get('msg')}")
        return d.get('data', d)

    def klines(self, sym, interval, limit=100):
        d = self._get('/openApi/swap/v3/quote/klines', {
            'symbol': sym, 'interval': interval, 'limit': limit
        })
        df = pd.DataFrame(d, columns=['time','open','high','low','close','volume','_','_2'])
        df = df[['time','open','high','low','close','volume']].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df.set_index('time')

    def ticker(self, sym):
        d = self._get('/openApi/swap/v2/quote/ticker', {'symbol': sym})
        return d[0] if isinstance(d, list) else d

    def orderbook(self, sym, depth=20):
        return self._get('/openApi/swap/v2/quote/depth', {
            'symbol': sym, 'limit': depth
        })

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

    def set_leverage(self, sym, lev):
        try:
            self._post('/openApi/swap/v2/trade/leverage', {
                'symbol': sym, 'side': 'LONG', 'leverage': lev
            })
            self._post('/openApi/swap/v2/trade/leverage', {
                'symbol': sym, 'side': 'SHORT', 'leverage': lev
            })
        except Exception as e:
            log.warning(f"Kaldıraç {sym}: {e}")

    def place_order(self, sym, side, qty, sl=None, tp=None):
        path = '/openApi/swap/v2/trade/order/test' if self.paper else '/openApi/swap/v2/trade/order'
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
                        'stopPrice': price, 'quoteOrderQty': qty,
                        'closePosition': True
                    })
                except Exception as e:
                    log.warning(f"Koşullu emir: {e}")
        return r


# ─────────────────────────────────────────────
#  TEKNİK ANALİZ ARAÇLARI
# ─────────────────────────────────────────────
class TA:
    @staticmethod
    def ema(arr, period):
        k = 2 / (period + 1)
        v = float(arr[0])
        for x in arr[1:]:
            v = float(x) * k + v * (1 - k)
        return v

    @staticmethod
    def ema_arr(arr, period):
        k = 2 / (period + 1)
        result = [float(arr[0])]
        for x in arr[1:]:
            result.append(float(x) * k + result[-1] * (1 - k))
        return np.array(result)

    @staticmethod
    def atr(df, period=14):
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def adx(df, period=14):
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        plus_dm  = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1], 0), 0)
        minus_dm = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:], 0), 0)
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        atr_val  = np.mean(tr[-period:]) + 1e-9
        plus_di  = 100 * np.mean(plus_dm[-period:])  / atr_val
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr_val
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        return float(dx), float(plus_di), float(minus_di)

    @staticmethod
    def supertrend(df, period=10, mult=3.0):
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        atr_vals = []
        for i in range(1, len(c)):
            tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            atr_vals.append(tr)
        atr_arr = np.array(atr_vals)
        upper = (h[1:]+l[1:])/2 + mult * atr_arr
        lower = (h[1:]+l[1:])/2 - mult * atr_arr
        trend = np.ones(len(c)-1)
        for i in range(1, len(trend)):
            if c[i] > upper[i-1]:
                trend[i] = 1
            elif c[i] < lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        return int(trend[-1])  # 1=UP, -1=DOWN

    @staticmethod
    def cvd(df):
        c = df['close'].values
        v = df['volume'].values
        o = df['open'].values
        delta = np.where(c > o, v, np.where(c < o, -v, 0))
        cvd_arr = np.cumsum(delta)
        recent  = float(np.mean(cvd_arr[-5:]))
        prev    = float(np.mean(cvd_arr[-10:-5]))
        return recent, recent > 0, recent > prev

    @staticmethod
    def rsi(arr, period=14):
        if len(arr) < period + 1:
            return 50.0
        d = np.diff(arr[-period-1:])
        g = d[d > 0].sum() / period
        l = -d[d < 0].sum() / period
        return 100.0 if l == 0 else 100 - 100 / (1 + g / l)

    @staticmethod
    def breakout(df, lookback=20):
        h = df['high'].values
        c = df['close'].values
        l = df['low'].values
        prev_high = np.max(h[-lookback-1:-1])
        prev_low  = np.min(l[-lookback-1:-1])
        if c[-1] > prev_high:
            return 'LONG'
        elif c[-1] < prev_low:
            return 'SHORT'
        return None

    @staticmethod
    def volume_spike(df, mult=2.0):
        v   = df['volume'].values
        avg = np.mean(v[-20:-1])
        return float(v[-1]) > avg * mult, float(v[-1]) / (avg + 1e-9)

    @staticmethod
    def htf_trend(df_daily, df_4h):
        c_d  = df_daily['close'].values
        c_4h = df_4h['close'].values

        ema50_d  = TA.ema(c_d, 50)
        ema200_d = TA.ema(c_d, 200)
        st_d     = TA.supertrend(df_daily)
        st_4h    = TA.supertrend(df_4h)
        adx_val, plus_di, minus_di = TA.adx(df_4h)

        daily_bull = ema50_d > ema200_d and c_d[-1] > ema50_d
        daily_bear = ema50_d < ema200_d and c_d[-1] < ema50_d

        if daily_bull and st_d == 1 and st_4h == 1 and adx_val > CONFIG['adx_threshold']:
            return 'BULL', adx_val
        elif daily_bear and st_d == -1 and st_4h == -1 and adx_val > CONFIG['adx_threshold']:
            return 'BEAR', adx_val
        return 'NEUTRAL', adx_val

    @staticmethod
    def obi(orderbook):
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            bid_vol = sum(float(b[1]) for b in bids[:10])
            ask_vol = sum(float(a[1]) for a in asks[:10])
            ratio   = bid_vol / (ask_vol + 1e-9)
            return float(ratio)
        except:
            return 1.0

    @staticmethod
    def liquidation_support(df, side='LONG'):
        c = df['close'].values
        l = df['low'].values
        h = df['high'].values
        atr_val = TA.atr(df)
        price   = c[-1]
        if side == 'LONG':
            support = price - atr_val * 1.5
            lows    = l[-20:]
            cluster = np.sum((lows >= support) & (lows <= price)) / 20
            return cluster > 0.3
        else:
            resist  = price + atr_val * 1.5
            highs   = h[-20:]
            cluster = np.sum((highs <= resist) & (highs >= price)) / 20
            return cluster > 0.3

    @staticmethod
    def big_order_detect(orderbook, threshold_usdt=50000):
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            big_bids = sum(1 for b in bids if float(b[0]) * float(b[1]) > threshold_usdt)
            big_asks = sum(1 for a in asks if float(a[0]) * float(a[1]) > threshold_usdt)
            return big_bids, big_asks
        except:
            return 0, 0


# ─────────────────────────────────────────────
#  ÇIKIŞ ANALİZİ
# ─────────────────────────────────────────────
class ExitAnalyzer:
    def __init__(self):
        self.oi_history = {}  # symbol → [oi değerleri]

    def update_oi(self, sym, oi_val):
        if sym not in self.oi_history:
            self.oi_history[sym] = []
        self.oi_history[sym].append(oi_val)
        if len(self.oi_history[sym]) > 10:
            self.oi_history[sym].pop(0)

    def oi_dropping(self, sym):
        hist = self.oi_history.get(sym, [])
        if len(hist) < 3:
            return False
        recent = np.mean(hist[-2:])
        prev   = np.mean(hist[-4:-2])
        change = (recent - prev) / (prev + 1e-9) * 100
        return change < CONFIG['oi_drop_pct']

    def cvd_divergence(self, df, side):
        c   = df['close'].values
        cvd_val, cvd_pos, cvd_rising = TA.cvd(df)
        if side == 'LONG':
            price_up = c[-1] > c[-5]
            return price_up and not cvd_rising
        else:
            price_dn = c[-1] < c[-5]
            return price_dn and cvd_rising

    def obi_reversed(self, orderbook, side):
        ratio = TA.obi(orderbook)
        if side == 'LONG':
            return ratio < 0.8
        else:
            return ratio > 1.2

    def htf_weakening(self, df_daily, df_4h):
        adx_val, _, _ = TA.adx(df_4h)
        st_4h = TA.supertrend(df_4h)
        st_d  = TA.supertrend(df_daily)
        return adx_val < 20 or (st_4h != st_d)

    def volatility_explosion(self, df):
        atr_now = TA.atr(df, 5)
        atr_avg = TA.atr(df, 20)
        return atr_now > atr_avg * CONFIG['atr_explosion_mult']

    def funding_extreme(self, funding):
        return funding > 0.05 or funding < -0.03

    def btc_crash(self, btc_df):
        c = btc_df['close'].values
        change = (c[-1] - c[-4]) / c[-4]
        return change < -0.02

    def position_too_old(self, opened_at):
        age = (datetime.now() - opened_at).seconds / 3600
        return age > CONFIG['max_position_age']

    def score(self, sym, side, df_1h, df_daily, df_4h, orderbook, funding, btc_df, opened_at):
        signals = 0
        reasons = []

        if self.oi_dropping(sym):
            signals += 1; reasons.append('OI↓')
        if self.cvd_divergence(df_1h, side):
            signals += 2; reasons.append('CVD_DIV')
        if self.obi_reversed(orderbook, side):
            signals += 1; reasons.append('OBI_REV')
        if self.htf_weakening(df_daily, df_4h):
            signals += 1; reasons.append('HTF_WEAK')
        if self.funding_extreme(funding):
            signals += 2; reasons.append('FUND_EXT')
        if self.btc_crash(btc_df):
            signals += 3; reasons.append('BTC_CRASH')
        if self.volatility_explosion(df_1h):
            signals += 1; reasons.append('VOL_EXP')
        if self.position_too_old(opened_at):
            signals += 1; reasons.append('TOO_OLD')

        if signals >= 4:
            return 'FULL_EXIT', reasons
        elif signals >= 2:
            return 'PARTIAL_EXIT', reasons
        elif signals >= 1:
            return 'TIGHTEN_SL', reasons
        return 'HOLD', reasons


# ─────────────────────────────────────────────
#  PAPER TRADING TRACKER
# ─────────────────────────────────────────────
class PaperTracker:
    def __init__(self, balance):
        self.balance   = balance
        self.start_bal = balance
        self.positions = {}
        self.trades    = []

    def open(self, sym, side, qty, entry, sl, tp1, tp2):
        if self.balance < qty:
            log.warning(f"Yetersiz bakiye: {self.balance:.2f}")
            return False
        self.positions[sym] = {
            'side': side, 'entry': entry, 'qty': qty,
            'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'tp1_hit': False, 'opened': datetime.now(),
            'partial': False
        }
        self.balance -= qty
        log.info(f"[PAPER] OPEN {side} {sym} @{entry:.4f} │ {qty} USDT")
        return True

    def partial_close(self, sym, price, reason):
        if sym not in self.positions:
            return 0
        pos  = self.positions[sym]
        half = pos['qty'] / 2
        pnl  = half * (price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
        self.balance    += half + pnl
        pos['qty']      -= half
        pos['partial']   = True
        pos['sl']        = pos['entry']
        self.trades.append({'sym': sym, 'pnl': pnl, 'reason': f'PARTIAL_{reason}'})
        log.info(f"[PAPER] PARTIAL {sym} @{price:.4f} │ PNL:{pnl:+.2f}")
        return pnl

    def close(self, sym, price, reason):
        if sym not in self.positions:
            return 0
        pos  = self.positions.pop(sym)
        pnl  = pos['qty'] * (price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
        self.balance += pos['qty'] + pnl
        self.trades.append({'sym': sym, 'pnl': pnl, 'reason': reason})
        em = '✅' if pnl > 0 else '❌'
        log.info(f"[PAPER] CLOSE {em} {sym} @{price:.4f} │ {reason} │ PNL:{pnl:+.2f}")
        return pnl

    def check_tp_sl(self, sym, price):
        if sym not in self.positions:
            return None, 0
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
            return 'SL', self.close(sym, price, 'SL')
        if hit_tp2:
            return 'TP2', self.close(sym, price, 'TP2')
        if hit_tp1:
            pos['tp1_hit'] = True
            pos['sl'] = pos['entry']
            log.info(f"[PAPER] TP1 {sym} → SL başa çekildi")
            return 'TP1', 0
        return None, 0

    def stats(self):
        pnls    = [t['pnl'] for t in self.trades]
        winning = [p for p in pnls if p > 0]
        losing  = [p for p in pnls if p <= 0]
        return {
            'count'    : len(pnls),
            'winning'  : len(winning),
            'losing'   : len(losing),
            'total_pnl': sum(pnls),
            'win_rate' : len(winning) / len(pnls) * 100 if pnls else 0,
            'balance'  : self.balance,
            'trades'   : self.trades
        }


# ─────────────────────────────────────────────
#  BİLDİRİM
# ─────────────────────────────────────────────
class Notifier:
    def __init__(self, channel):
        self.ch = channel
        self.ok = bool(channel)

    def _coin_to_cmc(self, sym):
        """BTC-USDT → CoinMarketCap linki"""
        name = sym.replace('-USDT','').replace('-','').lower()
        names = {
            'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana',
            'bnb': 'bnb', 'avax': 'avalanche-2', 'link': 'chainlink',
            'doge': 'dogecoin', 'arb': 'arbitrum', 'op': 'optimism',
            'pol': 'polygon', 'apt': 'aptos', 'sui': 'sui'
        }
        slug = names.get(name, name)
        return f"https://coinmarketcap.com/currencies/{slug}/"

    def _coin_to_bingx(self, sym):
        """BTC-USDT → BingX futures linki"""
        pair = sym.replace('-', '_')
        return f"https://bingx.com/en/futures/{pair}/"

    def send(self, msg, link=None, title=None, priority='default'):
        if not self.ok:
            log.info(f"[NOTIF] {msg[:100]}")
            return
        try:
            clean = msg.replace('<b>','').replace('</b>','').replace('<code>','').replace('</code>','')
            headers = {'Priority': priority}
            if title:
                headers['Title'] = title
            if link:
                headers['Click'] = link          # Bildirime tıklayınca açılır
                headers['Actions'] = (
                    f"view, 📊 CoinMarketCap, {self._coin_to_cmc(link) if 'BTC' in link or 'ETH' in link or 'SOL' in link or 'BNB' in link else link}; "
                    f"view, 🔄 BingX, {self._coin_to_bingx(link) if '-USDT' in link else link}"
                )
            requests.post(
                f'https://ntfy.sh/{self.ch}',
                data=clean.encode('utf-8'),
                headers=headers,
                timeout=5
            )
        except:
            pass

    def _ntfy_send(self, title, msg, sym, priority='default'):
        if not self.ok:
            log.info(f"[NOTIF] {title} | {msg[:80]}")
            return
        try:
            cmc   = self._coin_to_cmc(sym)
            bingx = self._coin_to_bingx(sym)
            headers = {
                'Title'   : title,
                'Priority': priority,
                'Click'   : cmc,
                'Actions' : f"view, CoinMarketCap, {cmc}; view, BingX Futures, {bingx}",
                'Tags'    : 'chart_with_upwards_trend'
            }
            requests.post(f'https://ntfy.sh/{self.ch}', data=msg.encode('utf-8'), headers=headers, timeout=5)
        except Exception as e:
            log.warning(f"Bildirim hatasi: {e}")
            try:
                requests.post(f'https://ntfy.sh/{self.ch}', data=msg.encode('utf-8'), timeout=5)
            except:
                pass

    def entry(self, sym, side, price, qty, sl, tp1, tp2, reasons, paper):
        mode  = 'PAPER' if paper else 'GERCEK'
        title = f"{sym} {side} ACILDI [{mode}]"
        msg   = (
            f"{'LONG' if side == 'LONG' else 'SHORT'} | {sym}\n"
            f"Fiyat: {price:.4f}\n"
            f"TP1:{tp1:.4f}  TP2:{tp2:.4f}\n"
            f"SL:{sl:.4f}\n"
            f"{qty} USDT\n"
            f"OK: {' | '.join(reasons)}"
        )
        self._ntfy_send(title, msg, sym, priority='high')

    def exit_msg(self, sym, reason, pnl, balance, paper):
        mode  = 'PAPER' if paper else 'GERCEK'
        kar   = 'KAR' if pnl > 0 else 'ZARAR'
        title = f"{sym} KAPANDI {kar} {pnl:+.2f} USDT [{mode}]"
        msg   = (
            f"{'KAZANC' if pnl > 0 else 'KAYIP'} | {sym}\n"
            f"Sebep: {reason}\n"
            f"PNL: {pnl:+.2f} USDT\n"
            f"Bakiye: {balance:.2f} USDT"
        )
        priority = 'high' if pnl > 0 else 'default'
        self._ntfy_send(title, msg, sym, priority=priority)

    def daily_report(self, stats, paper):
        mode = '📝 PAPER' if paper else '💰 GERÇEK'
        pnl  = stats['total_pnl']
        self.send(
            f"{'📈' if pnl >= 0 else '📉'} GÜNLÜK RAPOR [{mode}]\n"
            f"İşlem: {stats['count']}  ✅{stats['winning']} ❌{stats['losing']}\n"
            f"Win Rate: %{stats['win_rate']:.1f}\n"
            f"PNL: {pnl:+.2f} USDT\n"
            f"Bakiye: {stats['balance']:.2f} USDT"
        )

    def blocked(self, sym, missing):
        self.send(f"⚠️ {sym} — Koşullar sağlanmadı\n❌ {' │ '.join(missing)}")


# ─────────────────────────────────────────────
#  ANA BOT
# ─────────────────────────────────────────────
class CipherV2:
    def __init__(self):
        self.cfg    = CONFIG
        self.cli    = BingXClient(self.cfg['api_key'], self.cfg['api_secret'], self.cfg['paper_trading'])
        self.ta     = TA()
        self.exit_a = ExitAnalyzer()
        self.paper  = PaperTracker(self.cfg['initial_balance'])
        self.notif  = Notifier(self.cfg['ntfy_channel'])
        self.running = True

        # Risk takip
        self.daily_losses    = 0
        self.daily_pnl       = 0.0
        self.last_reset_date = date.today()
        self.sl_times        = {}  # symbol → son SL zamanı

        # Cache
        self.btc_cache = {}
        self.oi_prev   = {}

        log.info("═" * 50)
        log.info("  CIPHER V2 BAŞLADI")
        log.info(f"  Mod    : {'📝 PAPER' if self.cfg['paper_trading'] else '💰 GERÇEK'}")
        log.info(f"  Coinler: {', '.join(self.cfg['symbols'])}")
        log.info(f"  Kaldıraç: {self.cfg['leverage']}x")
        log.info("═" * 50)

        self.notif.send(
            f"🚀 CIPHER V2 BAŞLADI\n"
            f"{'📝 Paper' if self.cfg['paper_trading'] else '💰 GERÇEK'}\n"
            f"Coinler: {', '.join(self.cfg['symbols'])}\n"
            f"Kaldıraç: {self.cfg['leverage']}x\n"
            f"Bakiye: {self.cfg['initial_balance']} USDT"
        )

    # ── Günlük reset ──────────────────────────
    def _daily_reset(self):
        today = date.today()
        if today > self.last_reset_date:
            stats = self.paper.stats()
            self.notif.daily_report(stats, self.cfg['paper_trading'])
            self.daily_losses    = 0
            self.daily_pnl       = 0.0
            self.last_reset_date = today
            log.info("🔄 Günlük sayaçlar sıfırlandı")

    # ── BTC verisi cache ───────────────────────
    def _btc_data(self):
        now = time.time()
        if 'ts' in self.btc_cache and now - self.btc_cache['ts'] < 300:
            return self.btc_cache['d']
        d = {
            '1h'   : self.cli.klines('BTC-USDT', '1h',  200),
            '4h'   : self.cli.klines('BTC-USDT', '4h',  200),
            'daily': self.cli.klines('BTC-USDT', '1d',  200)
        }
        self.btc_cache = {'ts': now, 'd': d}
        return d

    # ── Dinamik pozisyon büyüklüğü ─────────────
    def _qty(self, score):
        if score >= 8:   return self.cfg['usdt_per_trade'] * 1.5
        elif score >= 6: return self.cfg['usdt_per_trade']
        else:            return self.cfg['usdt_per_trade'] * 0.7

    # ── Giriş analizi ──────────────────────────
    def _entry_check(self, sym, btc):
        passed  = []
        failed  = []
        score   = 0

        try:
            df_1h    = self.cli.klines(sym, '1h',  200)
            df_4h    = self.cli.klines(sym, '4h',  200)
            df_daily = self.cli.klines(sym, '1d',  200)
            df_15m   = self.cli.klines(sym, '15m', 100)
            ob       = self.cli.orderbook(sym, 20)
            oi       = self.cli.open_interest(sym)
            funding  = self.cli.funding_rate(sym)
        except Exception as e:
            log.warning(f"Veri hatası {sym}: {e}")
            return None

        c = df_1h['close'].values

        # ── 1. HTF Trend ──
        trend, adx_val = TA.htf_trend(df_daily, df_4h)
        if trend == 'BULL':
            passed.append(f'HTF_BULL(ADX:{adx_val:.0f})')
            score += 2
            side = 'LONG'
        elif trend == 'BEAR':
            passed.append(f'HTF_BEAR(ADX:{adx_val:.0f})')
            score += 2
            side = 'SHORT'
        else:
            failed.append('HTF_NEUTRAL')
            return None  # Trend belirsizse hiç işlem açma

        # ── 2. Multi TF Konsensüs ──
        st_15m = TA.supertrend(df_15m)
        st_1h  = TA.supertrend(df_1h)
        st_4h  = TA.supertrend(df_4h)
        exp    = 1 if side == 'LONG' else -1
        if st_15m == exp and st_1h == exp and st_4h == exp:
            passed.append('MTF_OK')
            score += 2
        else:
            failed.append(f'MTF_FAIL({st_15m}/{st_1h}/{st_4h})')
            return None

        # ── 3. Breakout ──
        bo = TA.breakout(df_1h)
        if bo == side:
            passed.append(f'BREAKOUT_{bo}')
            score += 1
        else:
            failed.append('NO_BREAKOUT')

        # ── 4. Volume Spike ──
        vol_spike, vol_ratio = TA.volume_spike(df_1h, self.cfg['volume_spike_mult'])
        if vol_spike:
            passed.append(f'VOL_SPIKE({vol_ratio:.1f}x)')
            score += 1
        else:
            failed.append(f'VOL_LOW({vol_ratio:.1f}x)')

        # ── 5. OI Artışı ──
        prev_oi = self.oi_prev.get(sym, oi)
        oi_chg  = (oi - prev_oi) / (prev_oi + 1e-9) * 100
        self.oi_prev[sym] = oi
        self.exit_a.update_oi(sym, oi)
        if oi_chg >= self.cfg['oi_change_pct']:
            passed.append(f'OI_UP({oi_chg:+.1f}%)')
            score += 1
        else:
            failed.append(f'OI_FLAT({oi_chg:+.1f}%)')

        # ── 6. Funding Rate ──
        f_ok = self.cfg['funding_min'] <= funding <= self.cfg['funding_max']
        if f_ok:
            passed.append(f'FUND_OK({funding:.4f})')
            score += 1
        else:
            failed.append(f'FUND_BAD({funding:.4f})')
            return None

        # ── 7. OBI ──
        obi_val = TA.obi(ob)
        if side == 'LONG' and obi_val >= self.cfg['obi_threshold']:
            passed.append(f'OBI_BULL({obi_val:.2f})')
            score += 1
        elif side == 'SHORT' and obi_val <= (1 / self.cfg['obi_threshold']):
            passed.append(f'OBI_BEAR({obi_val:.2f})')
            score += 1
        else:
            failed.append(f'OBI_NEUTRAL({obi_val:.2f})')

        # ── 8. CVD ──
        cvd_val, cvd_pos, cvd_rising = TA.cvd(df_1h)
        cvd_ok = (side == 'LONG' and cvd_pos and cvd_rising) or \
                 (side == 'SHORT' and not cvd_pos and not cvd_rising)
        if cvd_ok:
            passed.append(f'CVD_OK({cvd_val:.0f})')
            score += 1
        else:
            failed.append(f'CVD_FAIL({cvd_val:.0f})')

        # ── 9. Liquidation Desteği ──
        liq_ok = TA.liquidation_support(df_1h, side)
        if liq_ok:
            passed.append('LIQ_SUPPORT')
            score += 1
        else:
            failed.append('NO_LIQ_SUPPORT')

        # ── 10. Büyük Emir ──
        big_bids, big_asks = TA.big_order_detect(ob)
        if side == 'LONG' and big_bids > 0:
            passed.append(f'BIG_BID({big_bids})')
            score += 1
        elif side == 'SHORT' and big_asks > 0:
            passed.append(f'BIG_ASK({big_asks})')
            score += 1
        else:
            failed.append('NO_BIG_ORDER')

        # ── 11. BTC Korelasyon ──
        btc_c  = btc['1h']['close'].values
        btc_chg = (btc_c[-1] - btc_c[-4]) / btc_c[-4]
        if side == 'LONG' and btc_chg > self.cfg['btc_drop_limit']:
            passed.append(f'BTC_OK({btc_chg:+.2%})')
            score += 1
        elif side == 'SHORT':
            passed.append(f'BTC_SHORT_OK')
            score += 1
        else:
            failed.append(f'BTC_DROP({btc_chg:+.2%})')
            return None

        # ── Minimum skor kontrolü ──
        if score < 6:
            log.debug(f"  {sym} skor düşük: {score}/13 │ ❌ {', '.join(failed[:3])}")
            return None

        # ── TP / SL hesapla ──
        price   = float(c[-1])
        atr_val = TA.atr(df_1h)
        if side == 'LONG':
            sl  = price - atr_val * 1.2
            tp1 = price + atr_val * 1.5
            tp2 = price + atr_val * 3.0
        else:
            sl  = price + atr_val * 1.2
            tp1 = price - atr_val * 1.5
            tp2 = price - atr_val * 3.0

        return {
            'sym': sym, 'side': side, 'price': price,
            'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'score': score, 'passed': passed, 'failed': failed,
            'funding': funding, 'df_1h': df_1h, 'df_4h': df_4h,
            'df_daily': df_daily, 'ob': ob
        }

    # ── Açık pozisyon izleme ───────────────────
    def _monitor_positions(self, btc):
        for sym in list(self.paper.positions.keys()):
            try:
                tk    = self.cli.ticker(sym)
                price = float(tk.get('lastPrice', 0))
                pos   = self.paper.positions.get(sym)
                if not pos:
                    continue

                # TP/SL kontrolü
                reason, pnl = self.paper.check_tp_sl(sym, price)
                if reason and reason != 'TP1':
                    if pnl < 0:
                        self.daily_losses += 1
                    self.daily_pnl += pnl
                    self.notif.exit_msg(sym, reason, pnl, self.paper.balance, self.cfg['paper_trading'])
                    if reason == 'SL':
                        self.sl_times[sym] = time.time()
                    continue

                # Kademeli çıkış analizi
                try:
                    df_1h    = self.cli.klines(sym, '1h',  100)
                    df_4h    = self.cli.klines(sym, '4h',  100)
                    df_daily = self.cli.klines(sym, '1d',  100)
                    ob       = self.cli.orderbook(sym, 20)
                    funding  = self.cli.funding_rate(sym)

                    exit_sig, reasons = self.exit_a.score(
                        sym, pos['side'], df_1h, df_daily, df_4h,
                        ob, funding, btc['1h'], pos['opened']
                    )

                    if exit_sig == 'FULL_EXIT':
                        pnl = self.paper.close(sym, price, f"EXIT:{','.join(reasons)}")
                        self.daily_pnl += pnl
                        self.notif.exit_msg(sym, f"ÇIKIŞ: {', '.join(reasons)}", pnl, self.paper.balance, self.cfg['paper_trading'])
                        log.info(f"  🚪 FULL EXIT {sym} │ {', '.join(reasons)}")

                    elif exit_sig == 'PARTIAL_EXIT' and not pos.get('partial'):
                        pnl = self.paper.partial_close(sym, price, ','.join(reasons))
                        self.daily_pnl += pnl
                        self.notif.send(f"⚠️ {sym} KISMİ ÇIKIŞ\n{', '.join(reasons)}\nPNL: {pnl:+.2f}")
                        log.info(f"  ⚠️  PARTIAL EXIT {sym} │ {', '.join(reasons)}")

                    elif exit_sig == 'TIGHTEN_SL':
                        log.info(f"  🔒 SL SIKIŞTIRILDI {sym} │ {', '.join(reasons)}")

                except Exception as e:
                    log.warning(f"Çıkış analizi {sym}: {e}")

            except Exception as e:
                log.warning(f"İzleme {sym}: {e}")

    # ── Ana tarama ─────────────────────────────
    def scan(self):
        self._daily_reset()

        log.info(f"\n{'─'*50}")
        log.info(f"🔍 TARAMA │ {datetime.now().strftime('%H:%M:%S')} │ Açık: {len(self.paper.positions)}")

        # Günlük limit kontrolü
        if self.daily_losses >= self.cfg['max_daily_losses']:
            log.info("🛑 Günlük SL limiti doldu, işlem yok")
            return
        if self.daily_pnl < -self.cfg['daily_loss_limit']:
            log.info("🛑 Günlük zarar limiti aşıldı, işlem yok")
            return

        # BTC verisi
        try:
            btc = self._btc_data()
        except Exception as e:
            log.error(f"BTC verisi: {e}")
            return

        # Açık pozisyonları izle
        self._monitor_positions(btc)

        # Max pozisyon kontrolü
        if len(self.paper.positions) >= self.cfg['max_positions']:
            log.info(f"Max pozisyon dolu ({self.cfg['max_positions']})")
            return

        # Her coin için analiz
        for sym in self.cfg['symbols']:
            if not self.running:
                break
            if sym in self.paper.positions:
                continue
            if len(self.paper.positions) >= self.cfg['max_positions']:
                break

            # Tekrar giriş engeli
            last_sl = self.sl_times.get(sym, 0)
            if time.time() - last_sl < self.cfg['reentry_wait_secs']:
                wait = int((self.cfg['reentry_wait_secs'] - (time.time() - last_sl)) / 60)
                log.info(f"  ⏳ {sym} → {wait}dk bekleniyor (SL sonrası)")
                continue

            log.info(f"  ▷ {sym} analiz ediliyor...")

            result = self._entry_check(sym, btc)

            if result:
                qty = self._qty(result['score'])
                log.info(
                    f"  ✅ {sym} {result['side']} │ "
                    f"Skor:{result['score']} │ "
                    f"Miktar:{qty} USDT\n"
                    f"     Geçen: {', '.join(result['passed'])}"
                )

                if self.cfg['paper_trading']:
                    ok = self.paper.open(
                        sym, result['side'], qty,
                        result['price'], result['sl'],
                        result['tp1'], result['tp2']
                    )
                    if ok:
                        self.notif.entry(
                            sym, result['side'], result['price'],
                            qty, result['sl'], result['tp1'], result['tp2'],
                            result['passed'], True
                        )
                else:
                    self.cli.set_leverage(sym, self.cfg['leverage'])
                    self.cli.place_order(
                        sym,
                        'BUY' if result['side'] == 'LONG' else 'SELL',
                        qty, sl=result['sl'], tp=result['tp1']
                    )
                    self.notif.entry(
                        sym, result['side'], result['price'],
                        qty, result['sl'], result['tp1'], result['tp2'],
                        result['passed'], False
                    )
            else:
                log.info(f"  ✕ {sym} → Koşullar sağlanmadı")

            time.sleep(1)

        # Özet
        s = self.paper.stats()
        log.info(
            f"📊 Açık:{len(self.paper.positions)} │ "
            f"Bakiye:{self.paper.balance:.2f} │ "
            f"PNL:{s['total_pnl']:+.2f} │ "
            f"WR:%{s['win_rate']:.0f}"
        )

    # ── Bot döngüsü ────────────────────────────
    def run(self):
        interval = self.cfg['scan_interval_minutes'] * 60
        while self.running:
            try:
                self.scan()
            except Exception as e:
                log.error(f"Tarama hatası: {e}")
                self.notif.send(f"⚠️ Hata: {e}")
            log.info(f"💤 {self.cfg['scan_interval_minutes']}dk bekleniyor...\n")
            time.sleep(interval)


# ─────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────
if __name__ == '__main__':
    bot = CipherV2()
    try:
        bot.run()
    except KeyboardInterrupt:
        s = bot.paper.stats()
        bot.notif.send(f"⏹ Durduruldu │ PNL:{s['total_pnl']:+.2f} USDT")
        log.info("Bot durduruldu")
