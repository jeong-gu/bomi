# import requests

# api_key = "YOUR_API_KEY"  # ← 발급받은 키로 바꾸세요
# url = "https://newsapi.org/v2/top-headlines"

# params = {
#     "country": "kr",        # 국가 코드 (한국: 'kr', 미국: 'us')
#     "category": "technology",  # 카테고리: business, entertainment, general, health, science, sports, technology
#     "pageSize": 5,          # 뉴스 5개만
#     "apiKey": api_key
# }

# response = requests.get(url, params=params)
# data = response.json()

# for article in data["articles"]:
#     print(f"[{article['source']['name']}] {article['title']}")
#     print(article["url"])
#     print("-" * 40)
import sqlite3

print("sqlite3 모듈 정상 작동합니다.")
print("버전:", sqlite3.sqlite_version)
