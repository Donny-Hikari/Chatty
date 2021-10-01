
import requests

from utils import load_settings

def get_reddit_news(params):
    assert params['user-agent'], "An unique user agent name is required by reddit"

    res = requests.get('https://www.reddit.com/r/news/hot.json',
        headers={
            "accept": "application/json",
            "user-agent": params['user-agent'],
        })
    listing = res.json()
    if 'error' in listing.keys():
        print(f"Error occurs when fetching reddit news. [{listing['error']}] {listing['message']}.")
        return 'Nothing.'

    news = []
    for v in listing['data']['children']:
        if v['data']['author'] == "NewsModTeam":
            continue
        news.append(str(len(news)+1) + ". " + v['data']['title'])
    return '\n'.join(news)

if __name__ == "__main__":
    settings = load_settings()
    print(get_reddit_news(settings['news-feed-agent']['params']))
