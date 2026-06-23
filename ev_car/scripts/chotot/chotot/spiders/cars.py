from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode
import json
import scrapy
from loguru import logger


class CarsSpider(scrapy.Spider):
    name = "cars"
    allowed_domains = ["xe.chotot.com"]
    start_urls = ["https://xe.chotot.com/mua-ban-oto-dien-sdfu4"]

    def parse(self, response):
        cars_href = response.xpath("//ul/div[@role='button']//a/@href").getall()

        if cars_href:
            for href in cars_href:
                yield response.follow(href, callback=self.parse_car_details)

            parsed_url = urlparse(response.url)
            query_params = parse_qs(parsed_url.query)

            current_page = int(query_params.get('page', ['1'])[0])
            logger.info(f"Current page: {current_page}")
            next_page = current_page + 1

            query_params['page'] = [str(next_page)]
            new_query = urlencode(query_params, doseq=True)
            next_page_url = parsed_url._replace(query=new_query).geturl()

            yield scrapy.Request(url=next_page_url, callback=self.parse)

        else:
            self.logger.info(f"Reached the end of pagination at: {response.url}")

    def parse_car_details(self, response):
        next_data_script = response.xpath('//script[@id="__NEXT_DATA__"]/text()').get()
        if next_data_script:
            json_data = json.loads(next_data_script)

            ad_view = json_data.get('props', {}).get('initialState', {}).get('adView', {})
            ad_info = ad_view.get('adInfo', {})

            ad = ad_info.get('ad', {})
            ad_params = ad_info.get('ad_params', {})
            parameters = ad_info.get('parameters', [])
            params = ad_info.get('params', [])

            list_time_ms = ad.get('list_time')
            exact_date_posted = None
            if list_time_ms:
                # Divide by 1000 because JavaScript timestamps are in milliseconds
                exact_date_posted = datetime.fromtimestamp(list_time_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

            seller_profile = ad_view.get('sellerProfileForSeo', {})
            seller_rating = seller_profile.get('rating', {})

            yield {
                'url': response.url,
                'exact_date_posted': exact_date_posted,
                'seller_rating': seller_rating,
                'ad': ad,
                'ad_params': ad_params,
                'parameters': parameters,
                'params': params,
            }
