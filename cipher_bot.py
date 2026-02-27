"""CIPHER BOT - Railway.app versiyonu"""
import hashlib, hmac, time, logging, os, requests, numpy as np, pandas as pd
from datetime import datetime
from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s │ %(levelname)-7s │ %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler()])
log = logging.getLogger('CIPHER')

class BingXClient:
    BASE='https://open-api.bingx.com'
    def __init__(self,k,s,paper=True):
        self.api_key=k; self.secret=s; self.paper=paper
        self.session=requests.Session()
        self.session.headers.update({'X-BX-APIKEY':k})
        log.info(f"BingX │ {'📝 PAPER' if paper else '💰 GERÇEK'}")
    def _sign(self,p):
        q='&'.join(f'{k}={v}' for k,v in sorted(p.items()))
        return hmac.new(self.secret.encode(),q.encode(),hashlib.sha256).hexdigest()
    def _get(self,path,p=None):
        p=p or {}; p['timestamp']=int(time.time()*1000); p['signature']=self._sign(p)
        r=self.session.get(f'{self.BASE}{path}',params=p,timeout=10); r.raise_for_status()
        d=r.json()
        if d.get('code',0)!=0: raise Exception(f"API:{d.get('msg')}")
        return d.get('data',d)
    def _post(self,path,p=None):
        p=p or {}; p['timestamp']=int(time.time()*1000); p['signature']=self._sign(p)
        r=self.session.post(f'{self.BASE}{path}',params=p,timeout=10); r.raise_for_status()
        d=r.json()
        if d.get('code',0)!=0: raise Exception(f"API:{d.get('msg')}")
        return d.get('data',d)
    def get_klines(self,sym,interval,limit=80):
        d=self._get('/openApi/swap/v3/quote/klines',{'symbol':sym,'interval':interval,'limit':limit})
        df=pd.DataFrame(d,columns=['time','open','high','low','close','volume','_','_2'])
        df=df[['time','open','high','low','close','volume']].astype(float)
        df['time']=pd.to_datetime(df['time'],unit='ms')
        return df.set_index('time')
    def get_ticker(self,sym):
        d=self._get('/openApi/swap/v2/quote/ticker',{'symbol':sym})
        return d[0] if isinstance(d,list) else d
    def set_leverage(self,sym,lev):
        try: self._post('/openApi/swap/v2/trade/leverage',{'symbol':sym,'side':'LONG','leverage':lev})
        except Exception as e: log.warning(f"Kaldıraç:{e}")
    def place_order(self,sym,side,qty,sl=None,tp=None):
        path='/openApi/swap/v2/trade/order/test' if self.paper else '/openApi/swap/v2/trade/order'
        r=self._post(path,{'symbol':sym,'side':side,'positionSide':'LONG' if side=='BUY' else 'SHORT','type':'MARKET','quoteOrderQty':qty})
        for price,otype in [(sl,'STOP_MARKET'),(tp,'TAKE_PROFIT_MARKET')]:
            if price:
                try:
                    cs='SELL' if side=='BUY' else 'BUY'
                    self._post(path,{'symbol':sym,'side':cs,'positionSide':'LONG' if side=='BUY' else 'SHORT','type':otype,'stopPrice':price,'quoteOrderQty':qty,'closePosition':True})
                except Exception as e: log.warning(f"Koşullu emir:{e}")
        return r

class Cipher:
    def _ret(self,p): return np.diff(p)/p[:-1]
    def _pc(self,a,b):
        n=min(len(a),len(b)); a,b=a[:n],b[:n]
        if np.std(a)==0 or np.std(b)==0: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    def _rsi(self,c,p=14):
        if len(c)<p+1: return 50.0
        d=np.diff(c[-p-1:]); g=d[d>0].sum()/p; l=-d[d<0].sum()/p
        return 100.0 if l==0 else 100-100/(1+g/l)
    def _ema(self,a,p):
        k=2/(p+1); v=a[0]
        for x in a[1:]: v=x*k+v*(1-k)
        return v
    def _bb(self,c,p=20):
        s=c[-p:]; m=np.mean(s); sd=np.std(s)
        return (c[-1]-(m-2*sd))/(4*sd+1e-9)
    def lg(self,bc,ac):
        br=self._ret(bc); ar=self._ret(ac)
        bm=(bc[-1]-bc[-10])/bc[-10]; am=(ac[-1]-ac[-10])/ac[-10]; dv=bm-am
        s=0.0
        if bm>0 and dv>0.01: s+=0.5
        if bm>0 and dv<-0.01: s-=0.3
        if bm<-0.01: s-=0.4
        if self._pc(br[-20:],ar[-20:])-self._pc(br[-40:-20],ar[-40:-20])>0.1: s+=0.2
        return s
    def ph(self,bc,ac):
        br=self._ret(bc); ar=self._ret(ac); bl,bco=0,-1
        for lag in range(9):
            c=self._pc(br[lag:],ar[:-lag] if lag else ar)
            if c>bco: bco=c; bl=lag
        un=float(np.mean(self._ret(bc[-6:])))*bco if bl>0 else 0
        return 0.7 if un>0.003 else -0.7 if un<-0.003 else 0.3 if un>0 else -0.3
    def en(self,df):
        c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
        bp=self._bb(c)
        vr=np.mean(v[-5:])/(np.mean(v[-20:])+1e-9)
        ar=np.mean(h[-5:]-l[-5:])/(np.mean(h[-20:]-l[-20:])+1e-9)
        co=1.0 if ar<0.8 else 0.5 if ar<1 else 0.0
        vp=1.0 if vr>1.3 else 0.5 if vr>1 else 0.0
        el=co*0.5+vp*0.5
        d=1 if bp>0.6 and vr>1 else -1 if bp<0.4 and vr>1 else 0
        return el*d, el
    def mr(self,df):
        c=df['close'].values; h=df['high'].values; l=df['low'].values
        atr=np.mean(h[-14:]-l[-14:]); atp=atr/c[-1]*100
        adx=min(100,abs(c[-1]-c[-14])/(atr+1e-9)*10)
        r=self._ret(c[-30:]); hv=[]
        for w in [4,8,16]:
            s=r[-w:]; rng=np.ptp(s); sd=np.std(s)
            if sd>0: hv.append(np.log(rng/sd)/np.log(w))
        h2=max(0,min(1,float(np.mean(hv)) if hv else 0.5))
        rg='TREND' if adx>25 and h2>0.55 else 'VOLATİL' if atp>4 else 'BANT'
        sc=0.5 if h2>0.55 else -0.2 if h2<0.45 else 0.1
        return sc, rg, atp, self._rsi(c)
    def tf(self,df,rg):
        c=df['close'].values; v=df['volume'].values
        rs=self._rsi(c); mc=self._ema(c,12)-self._ema(c,26); bp=self._bb(c)
        e9=self._ema(c,9); e21=self._ema(c,21)
        obv=0.0; oa=[0.0]
        for i in range(1,len(c)):
            obv+=v[i] if c[i]>c[i-1] else -v[i] if c[i]<c[i-1] else 0; oa.append(obv)
        ob=np.mean(oa[-5:])>np.mean(oa[-10:-5])
        vt=0.0
        if rs<35: vt+=1.5
        elif rs>65: vt-=1.5
        vt+=1.0 if mc>0 else -1.0
        if bp<0.2: vt+=1.2
        elif bp>0.8: vt-=1.2
        vt+=1.0 if e9>e21 else -1.0
        vt+=0.8 if ob else -0.8
        if rg=='TREND': vt*=1.2
        elif rg=='VOLATİL': vt*=0.7
        n=vt/5.5
        return 'LONG' if n>0.3 else 'SHORT' if n<-0.3 else 'WAIT', n
    def analyze(self,sym,b1h,d30,d1h,d4h):
        bc=b1h['close'].values; ac=d1h['close'].values
        lgs=self.lg(bc,ac); phs=self.ph(bc,ac); ens,_=self.en(d1h)
        mrs,rg,atp,rsi=self.mr(d1h)
        t30,_=self.tf(d30,rg); t1h,_=self.tf(d1h,rg); t4h,_=self.tf(d4h,rg)
        votes={'LONG':0.0,'SHORT':0.0,'WAIT':0.0}
        for tf,w in [(t30,0.2),(t1h,0.35),(t4h,0.45)]: votes[tf]+=w
        ms=lgs*0.3+phs*0.3+ens*0.25+mrs*0.15
        fs=(votes['LONG']-votes['SHORT'])*0.6+ms*0.4
        if fs>0.2: sig,conf='LONG',min(0.95,0.5+fs*0.5)
        elif fs<-0.2: sig,conf='SHORT',min(0.95,0.5+abs(fs)*0.5)
        else: sig,conf='WAIT',0.5
        uni=all(t==sig for t in [t30,t1h,t4h])
        if uni and sig!='WAIT': conf=min(0.97,conf+0.08)
        p=float(d1h['close'].iloc[-1]); a=atp/100*p
        return {'symbol':sym,'price':p,'signal':sig,'confidence':conf,'unanimous':uni,
                'tp1':p+a*1.5 if sig=='LONG' else p-a*1.5,
                'tp2':p+a*3.0 if sig=='LONG' else p-a*3.0,
                'sl':p-a if sig=='LONG' else p+a,
                'regime':rg,'rsi':rsi,'tf30':t30,'tf1h':t1h,'tf4h':t4h,
                'ts':datetime.now()}

class TG:
    def __init__(self,token,cid):
        self.token=token; self.cid=cid
        self.ok=bool(token and token not in ('','YOUR_TOKEN'))
    def send(self,msg):
        if not self.ok: log.info(f"[TG] {msg[:80]}"); return
        try: requests.post(f'https://api.telegram.org/bot{self.token}/sendMessage',json={'chat_id':self.cid,'text':msg,'parse_mode':'HTML'},timeout=5)
        except: pass
    def signal_msg(self,r,qty,paper):
        em='🟢' if r['signal']=='LONG' else '🔴'
        self.send(f"{em} <b>CIPHER {'PAPER' if paper else 'GERÇEK'}</b>\n"
                  f"📌 <b>{r['symbol']}</b> → <b>{r['signal']}</b>\n"
                  f"💵 Fiyat: <code>{r['price']:.4f}</code>\n"
                  f"💎 Güven: <code>%{int(r['confidence']*100)}</code>{'  ⬟3/3' if r['unanimous'] else ''}\n"
                  f"🕐 {r['tf30']}│{r['tf1h']}│{r['tf4h']}\n"
                  f"🎯 TP1:<code>{r['tp1']:.4f}</code>  TP2:<code>{r['tp2']:.4f}</code>\n"
                  f"🛑 SL:<code>{r['sl']:.4f}</code>\n"
                  f"💰 {qty} USDT │ ⏰{r['ts'].strftime('%H:%M')}")
    def report(self,trades,bal):
        if not trades: self.send("📊 Bugün işlem yok."); return
        pnl=sum(t['pnl'] for t in trades); w=sum(1 for t in trades if t['pnl']>0)
        self.send(f"{'📈' if pnl>=0 else '📉'} <b>GÜNLÜK RAPOR</b>\n"
                  f"İşlem:{len(trades)} ✅{w} ❌{len(trades)-w}\n"
                  f"PNL:<code>{pnl:+.2f}</code> USDT\n"
                  f"Bakiye:<code>{bal:.2f}</code> USDT")

class Paper:
    def __init__(self,bal): self.bal=bal; self.pos={}; self.trades=[]
    def open(self,sym,side,qty,entry,sl,tp1,tp2):
        if self.bal<qty: return False
        self.pos[sym]={'side':side,'entry':entry,'qty':qty,'sl':sl,'tp1':tp1,'tp2':tp2,'tp1_hit':False}
        self.bal-=qty; log.info(f"[PAPER] OPEN {side} {sym} @{entry:.4f} │{qty}USDT"); return True
    def check(self,sym,price):
        if sym not in self.pos: return None
        p=self.pos[sym]; s=p['side']
        if (s=='LONG' and price<=p['sl']) or (s=='SHORT' and price>=p['sl']): return self._cl(sym,price,'SL')
        if (s=='LONG' and price>=p['tp2']) or (s=='SHORT' and price<=p['tp2']): return self._cl(sym,price,'TP2')
        if not p['tp1_hit'] and ((s=='LONG' and price>=p['tp1']) or (s=='SHORT' and price<=p['tp1'])):
            p['tp1_hit']=True; p['sl']=p['entry']; log.info(f"[PAPER] TP1 {sym} → SL başa çekildi"); return ('TP1',None)
        return None
    def _cl(self,sym,price,reason):
        p=self.pos.pop(sym)
        pnl=p['qty']*(price-p['entry'])/p['entry']*(1 if p['side']=='LONG' else -1)
        self.bal+=p['qty']+pnl; self.trades.append({'pnl':pnl,'reason':reason})
        log.info(f"[PAPER] CLOSE {sym} @{price:.4f} │{reason}│PNL:{pnl:+.2f}"); return (reason,pnl)

class CipherBot:
    def __init__(self):
        self.cfg=CONFIG
        self.cli=BingXClient(self.cfg['api_key'],self.cfg['api_secret'],self.cfg['paper_trading'])
        self.ci=Cipher(); self.tg=TG(self.cfg['telegram_token'],self.cfg['telegram_chat_id'])
        self.pa=Paper(self.cfg['initial_balance']); self.bc={}; self.running=True
        log.info("═"*45+"\n  CIPHER BOT BAŞLADI\n"+"═"*45)
        self.tg.send(f"🚀 <b>CIPHER BOT BAŞLADI</b>\n"
                     f"{'📝 Paper' if self.cfg['paper_trading'] else '💰 GERÇEK'}\n"
                     f"Coinler: {len(self.cfg['symbols'])} │ Güven: %{int(self.cfg['min_confidence']*100)}\n"
                     f"Kaldıraç: {self.cfg['leverage']}x │ Bakiye: {self.cfg['initial_balance']} USDT")
    def _btc(self):
        now=time.time()
        if 'ts' in self.bc and now-self.bc['ts']<300: return self.bc['d']
        d={'1h':self.cli.get_klines('BTC-USDT','1h',80)}
        self.bc={'ts':now,'d':d}; return d
    def scan(self):
        log.info(f"\n{'─'*40}\n🔍 {datetime.now().strftime('%H:%M:%S')}")
        for sym in list(self.pa.pos.keys()):
            try:
                tk=self.cli.get_ticker(sym); price=float(tk.get('lastPrice',0))
                res=self.pa.check(sym,price)
                if res and res[0]!='TP1' and res[1] is not None:
                    em='✅' if res[1]>0 else '❌'
                    self.tg.send(f"{em} <b>{sym}</b> {res[0]}\nPNL:<code>{res[1]:+.2f}</code> │ Bakiye:<code>{self.pa.bal:.2f}</code>")
            except Exception as e: log.warning(f"İzleme {sym}:{e}")
        if len(self.pa.pos)>=self.cfg['max_positions']:
            log.info(f"Max pozisyon ({self.cfg['max_positions']}) dolu"); return
        try: btc=self._btc()
        except Exception as e: log.error(f"BTC:{e}"); return
        for raw in self.cfg['symbols']:
            if not self.running: break
            sym=raw if '-' in raw else f"{raw}-USDT"
            if sym in self.pa.pos or len(self.pa.pos)>=self.cfg['max_positions']: continue
            try:
                d30=self.cli.get_klines(sym,'30m',80)
                d1h=self.cli.get_klines(sym,'1h',80)
                d4h=self.cli.get_klines(sym,'4h',80)
                r=self.ci.analyze(sym,btc['1h'],d30,d1h,d4h)
                log.info(f"  {sym:14} {r['signal']:5} %{int(r['confidence']*100)} {'⬟' if r['unanimous'] else ' '} {r['regime']} RSI:{r['rsi']:.0f}")
                if r['signal']=='LONG' and r['confidence']>=self.cfg['min_confidence']:
                    qty=self.cfg['usdt_per_trade']
                    if self.cfg['paper_trading']:
                        if self.pa.open(sym,'LONG',qty,r['price'],r['sl'],r['tp1'],r['tp2']):
                            self.tg.signal_msg(r,qty,True)
                    else:
                        self.cli.set_leverage(sym,self.cfg['leverage'])
                        self.cli.place_order(sym,'BUY',qty,sl=r['sl'],tp=r['tp1'])
                        self.tg.signal_msg(r,qty,False)
                time.sleep(0.8)
            except Exception as e: log.warning(f"  ✕ {sym}:{e}"); time.sleep(1)
        s={'pnl':sum(t['pnl'] for t in self.pa.trades)}
        log.info(f"📊 Açık:{len(self.pa.pos)} │ Bakiye:{self.pa.bal:.2f} │ PNL:{s['pnl']:+.2f}")
    def run(self):
        interval=self.cfg['scan_interval_minutes']*60; last=datetime.now().date()
        while self.running:
            try: self.scan()
            except Exception as e: log.error(f"Hata:{e}"); self.tg.send(f"⚠️ Hata:{e}")
            if datetime.now().date()>last:
                self.tg.report(self.pa.trades,self.pa.bal); last=datetime.now().date()
            log.info(f"💤 {self.cfg['scan_interval_minutes']}dk bekleniyor...")
            time.sleep(interval)

if __name__=='__main__':
    bot=CipherBot()
    try: bot.run()
    except KeyboardInterrupt:
        pnl=sum(t['pnl'] for t in bot.pa.trades)
        bot.tg.send(f"⏹ Durduruldu │ PNL:{pnl:+.2f} USDT")
