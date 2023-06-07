from requests_html import HTMLSession
import json
import time
import random
import re

class Reviews:
    # asin = Amazon Standard Identification Number (sheesh alam ni copilot, product identifier for amazon)
    def __init__(self, url) -> None:
        # self.asin = asin
        self.url = url
        self.session = HTMLSession()
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}
        
    def pagination(self, page):
        page_number = str(page)
        if page > 1:
            button_url = "ref=cm_cr_arp_d_paging_btm_next_{}?".format(page)
            new_url = re.sub(r"ref=cm_cr_dp_d_show_all_btm?", button_url, self.url)
            r = self.session.get(new_url + '&pageNumber=' + page_number, headers=self.headers)
        else:
            new_url = self.url
            r = self.session.get(new_url, headers=self.headers)
        timeout_interval = random.uniform(20, 21)
        r.html.render(timeout=timeout_interval)
        if not r.html.find('div[data-hook=review]'):
            return False
        else:
            return r.html.find('div[data-hook=review]')

    def parse(self, reviews):
        total = []
        for review in reviews:
            # name = review.find('.a-profile-name', first=True).text
            # title = review.find('a[data-hook=review-title]', first=True).text
            # date = review.find('span[data-hook=review-date]', first=True).text
            # rating = review.find('i[data-hook=review-star-rating] span', first=True).text
            body = review.find('span[data-hook=review-body]', first=True).text

            # data = {
            #     'name': name,
            #     'date': date,
            #     'title': title,
            #     'rating': rating,
            #     'body': body
            # }
            # total.append(data)
            total.append(body)
        return total

def scrape(url):
    scraper = Reviews(url)
    results = []
    # x = 1
    for x in range(1, 21):
        reviews = scraper.pagination(x)
        print('Page found: ', x)
        if reviews is not False:
            results.extend(scraper.parse(reviews))
            # x += 1
        else:
            print(f'Traversed {x-1} pages')
            print('No more reviews')
            break
        interval = random.uniform(10, 13)
        time.sleep(interval)
    # return json.dumps(results, indent=2)
    # print(json.dumps(results))
    results_count = len(results)
    print('Total reviews: ', results_count)
    with open('content.json', 'w') as f:
        json.dump(results, f, indent=4)
    # reviews = scraper.pagination(1)
    # print(scraper.parse(reviews))

print(scrape('https://www.amazon.com/VERSACE-Homme-Dylan-Toilette-Spray/product-reviews/B01JG5UT64/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'))
