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
    #  COİNLER — Tüm Majörler
    # ═══════════════════════════════════════
    'symbols': [
        'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'BNB-USDT',
        'AVAX-USDT', 'LINK-USDT', 'ARB-USDT', 'OP-USDT',
        'DOGE-USDT', 'APT-USDT', 'SUI-USDT', 'DOT-USDT',
        'POL-USDT', 'ATOM-USDT', 'INJ-USDT',
    ],

    # ═══════════════════════════════════════
    #  POZİSYON YÖNETİMİ
    # ═══════════════════════════════════════
    'leverage'          : 5,
    'entry1_usdt'       : 25.0,   # 1. kademe giriş
    'entry2_usdt'       : 75.0,   # 2. kademe giriş (trend onayı)
    'max_positions'     : 6,      # Max eş zamanlı pozisyon

    # ═══════════════════════════════════════
    #  GİRİŞ FİLTRELERİ
    # ═══════════════════════════════════════
    'adx_threshold'     : 20,     # ADX trend gücü (düşürüldü - daha fazla sinyal)
    'obi_threshold'     : 1.2,    # Order Book Imbalance
    'volume_spike_mult' : 1.5,    # Hacim çarpanı (düşürüldü)
    'oi_change_pct'     : 0.3,    # OI artış eşiği
    'funding_max'       : 0.05,
    'funding_min'       : -0.02,
    'btc_drop_limit'    : -0.025, # BTC bu kadar düşerse long açma

    # ═══════════════════════════════════════
    #  DESTEK/DİRENÇ BAZLI SL
    # ═══════════════════════════════════════
    'swing_lookback'    : 20,     # Swing high/low için geriye bakış
    'pivot_atr_buffer'  : 0.3,    # Pivot seviyesine buffer (ATR çarpanı)
    'ob_lookback'       : 50,     # Order block için geriye bakış
    'sl_min_pct'        : 0.015,  # Min SL mesafesi (%1.5)
    'sl_max_pct'        : 0.08,   # Max SL mesafesi (%8)

    # ═══════════════════════════════════════
    #  TP HEDEFLERİ
    # ═══════════════════════════════════════
    'tp1_rr'            : 1.5,    # Risk/Reward TP1
    'tp2_rr'            : 3.0,    # Risk/Reward TP2

    # ═══════════════════════════════════════
    #  ÇIKIŞ FİLTRELERİ
    # ═══════════════════════════════════════
    'oi_drop_pct'       : -0.5,
    'max_position_age'  : 6,      # Max pozisyon süresi (saat)
    'atr_explosion_mult': 2.5,

    # ═══════════════════════════════════════
    #  RİSK
    # ═══════════════════════════════════════
    'reentry_wait_secs' : 3600,   # SL/TP sonrası 1 saat bekle

    # ═══════════════════════════════════════
    #  TARAMA
    # ═══════════════════════════════════════
    'scan_interval_minutes': 15,
}
