# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ChototItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class CarItem(scrapy.Item):
    # Metadata
    url = scrapy.Field()
    ad_id = scrapy.Field()
    list_time = scrapy.Field()

    # Categorical & Numerical IDs (Great for ML)
    carbrand_id = scrapy.Field()
    carmodel_id = scrapy.Field()
    price = scrapy.Field()
    mileage_v2 = scrapy.Field()
    year = scrapy.Field()

    # Text Labels (Great for EDA / Displays)
    carbrand_name = scrapy.Field()
    carmodel_name = scrapy.Field()
    condition_name = scrapy.Field()

    # Location
    region_name = scrapy.Field()
    area_name = scrapy.Field()

    # Images (We will join the list into a single string separated by commas)
    image_urls = scrapy.Field()

    seller_id = scrapy.Field()
    seller_name = scrapy.Field()
