# url = ('https://newsapi.org/v2/top-headlines?'+
# 	'country=us&'+
# 	'apiKey=66954171c19248dc8097abd04cfc8a4a')
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='66954171c19248dc8097abd04cfc8a4a')
all_articles = newsapi.get_everything(q='NYSE',
                                      from_param='2019-02-11',
                                      to='2019-03-10',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

# response = requests.get(url)
print(all_articles)
