import os
CONFIG = {
    'api_key'    : os.environ.get('BINGX_API_KEY', 'YOUR_API_KEY'),
    'api_secret' : os.environ.get('BINGX_SECRET',  'YOUR_SECRET'),
    'telegram_token'   : os.environ.get('TG_TOKEN',   ''),
    'telegram_chat_id' : os.environ.get('TG_CHAT_ID', ''),
    'paper_trading'   : os.environ.get('PAPER_MODE', 'true').lower() == 'true',
    'initial_balance' : float(os.environ.get('START_BALANCE', '1000')),
    'symbols': [
        'BTC-USDT', 'ETH-USDT', 'SOL-USDT',
        'BNB-USDT', 'AVAX-USDT', 'DOGE-USDT',
        'ARB-USDT', 'OP-USDT', 'LINK-USDT',
        'POL-USDT', 'APT-USDT', 'SUI-USDT',
    ],
    'min_confidence' : float(os.environ.get('MIN_CONFIDENCE', '0.60')),
    'long_only'      : False,   # ← False = LONG + SHORT her iki yön
    'usdt_per_trade' : float(os.environ.get('TRADE_SIZE', '50')),
    'max_positions'  : int(os.environ.get('MAX_POS', '5')),
    'leverage'       : int(os.environ.get('LEVERAGE', '5')),
    'scan_interval_minutes': int(os.environ.get('SCAN_INTERVAL', '15')),
}
