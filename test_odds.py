
import requests, sys
sys.path.insert(0, '.')
from core.utils.config_loader import load_config

cfg = load_config(sport='mlb', base_dir='config')
key = cfg.get('ODDS_API_KEY')
print('API Key: ' + (key[:8] + '...' if key else 'NO CONFIGURADA'))

if key:
    r = requests.get('https://api.the-odds-api.com/v4/sports/baseball_mlb/odds',
        params={'apiKey': key, 'regions': 'us', 'markets': 'totals,h2h,spreads',
                'dateFormat': 'iso'},
        timeout=20)
    print('Status: ' + str(r.status_code))
    print('Creditos: ' + str(r.headers.get('x-requests-remaining', '?')))
    if r.status_code == 200:
        games = r.json()
        print('Juegos con odds: ' + str(len(games)))
        for g in games[:3]:
            print('  ' + g['away_team'] + ' @ ' + g['home_team'])
            for bm in g.get('bookmakers', [])[:1]:
                for mkt in bm.get('markets', []):
                    print('    ' + mkt['key'] + ': ' + str([o['price'] for o in mkt['outcomes'][:2]]))
    else:
        print('Error: ' + r.text[:200])