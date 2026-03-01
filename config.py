import os

CONFIG = {
    # ═══════════════════════════════════════
    #  BINGX API
    # ═══════════════════════════════════════
    'api_key'    : os.environ.get('BINGX_API_KEY', 'YOUR_API_KEY'),
    'api_secret' : os.environ.get('BINGX_SECRET',  'YOUR_SECRET'),

    # ═══════════════════════════════════════
    #  BİLDİRİM
    # ═══════════════════════════════════════
    'ntfy_channel' : os.environ.get('NTFY_CHANNEL', ''),

    # ═══════════════════════════════════════
    #  MOD
    # ═══════════════════════════════════════
    'paper_trading'   : os.environ.get('PAPER_MODE', 'true').lower() == 'true',
    'initial_balance' : float(os.environ.get('START_BALANCE', '1000')),

    # ═══════════════════════════════════════
    #  COİNLER
    # ═══════════════════════════════════════
    'symbols': [
        'BTC-USDT',
        'ETH-USDT',
        'SOL-USDT',
        'BNB-USDT',
    ],

    # ═══════════════════════════════════════
    #  POZİSYON YÖNETİMİ
    # ═══════════════════════════════════════
    'leverage'          : 5,
    'usdt_per_trade'    : 50.0,
    'max_positions'     : 4,

    # ═══════════════════════════════════════
    #  GİRİŞ FİLTRELERİ
    # ═══════════════════════════════════════
    'obi_threshold'     : 1.25,   # Order Book Imbalance eşiği
    'volume_spike_mult' : 2.0,    # Hacim ortalamanın kaç katı olmalı
    'oi_change_pct'     : 0.5,    # OI en az %0.5 artmalı
    'funding_max'       : 0.05,   # Max funding rate
    'funding_min'       : -0.01,  # Min funding rate
    'adx_threshold'     : 25,     # ADX trend gücü eşiği
    'btc_drop_limit'    : -0.02,  # BTC bu kadar düşerse long açma

    # ═══════════════════════════════════════
    #  ÇIKIŞ FİLTRELERİ
    # ═══════════════════════════════════════
    'oi_drop_pct'       : -0.5,   # OI bu kadar düşerse çıkış sinyali
    'max_position_age'  : 4,      # Max pozisyon süresi (saat)
    'atr_explosion_mult': 2.5,    # ATR bu kadar artarsa volatilite patlaması

    # ═══════════════════════════════════════
    #  RİSK YÖNETİMİ
    # ═══════════════════════════════════════
    'max_daily_losses'  : 3,      # Günde max SL sayısı
    'reentry_wait_secs' : 7200,   # SL sonrası bekleme (2 saat)
    'daily_loss_limit'  : 100.0,  # Günlük max zarar (USDT)

    # ═══════════════════════════════════════
    #  TARAMA
    # ═══════════════════════════════════════
    'scan_interval_minutes': 15,
}
