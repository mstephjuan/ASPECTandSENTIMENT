from requests_html import HTMLSession
import json
import time
import random
import re
from bs4 import BeautifulSoup

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
                # 'name': name,
                # 'date': date,
                # 'title': title,
                # 'rating': rating,
                # "body": body
            # }
            # total.append(data)
            total.append(body)
        return total
    
def scrape(url):
    scraper = Reviews(url)
    results = []
    # x = 1
    for x in range(1, 8):
        reviews = scraper.pagination(x)
        print('Page found: ', x)
        if reviews is not False:
            results.extend(scraper.parse(reviews))
            # x += 1
        else:
            print(f'Traversed {x-1} pages')
            print('No more reviews')
            break
        interval = random.uniform(3, 4)
        time.sleep(interval)
    return results
    # print(product)
    # print(json.dumps(results))
    # results_count = len(results)
    # print('Total reviews: ', results_count)
    # with open('content.json', 'w') as f:
    #     json.dump(results, f, indent=4)
    # reviews = scraper.pagination(1)
    # print(scraper.parse(reviews))

def getProductTitle(url):
    # Create an HTML session
    session = HTMLSession()

    # Send a GET request to the webpage
    response = session.get(url)  # Replace 'https://www.example.com' with the actual URL

    # Render the JavaScript on the webpage
    response.html.render()

    # Find the element using the specified CSS selector
    element = response.html.find('a[data-hook="product-link"].a-link-normal', first=True)

    # Extract the href and text content from the element
    text = element.text

    # Print the results
    print(f"Text: {text}")
    return text

# link = 'https://www.amazon.com/Sanabul-Womens-Easter-Boxing-Gloves/product-reviews/B08L87WGF4/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
# print(getProductTitle(link))
# print(scrape(link))