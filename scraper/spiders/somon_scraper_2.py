import scrapy
import re

class SomonSpider(scrapy.Spider):
    name = "somon_spider"
    allowed_domains = ["somon.tj"]

    # === User-defined filters (defaults) ===
    rooms = "3-komnatnyie"
    build_state = "sostoyanie---10"       # already built
   # property_type = "type---1"            # secondary market
    city = "hudzhand"

    def __init__(self, *args, **kwargs):
        super(SomonSpider, self).__init__(*args, **kwargs)
        
        # Override defaults with passed parameters
        self.rooms = kwargs.get('rooms', self.rooms)
        self.build_state = kwargs.get('build_state', self.build_state)
        self.city = kwargs.get('city', self.city)
        self.property_type = kwargs.get('property_type', None)
        
        self.logger.info(f"Spider initialized with: rooms={self.rooms}, build_state={self.build_state}, city={self.city}, property_type={self.property_type}")

    def start_requests(self):
        # Build URL with user parameters
        url_parts = [
            "https://somon.tj/nedvizhimost/prodazha-kvartir",
            self.rooms,
            self.build_state,
            self.city
        ]
        
        if self.property_type:
            url_parts.append(self.property_type)
            
        url = "/".join(url_parts) + "/"
        self.logger.info(f"Starting scrape with URL: {url}")
        yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Extract each ad block (using working selectors from somon_listings)
        ads = response.css("div.advert.js-item-listing")
        
        for ad in ads:
            # Extract the URL from the <a> with class 'advert__content-title' or the <a> with class 'mask'
            relative_url = ad.css("a.advert__content-title::attr(href), a.mask::attr(href)").get()
            if relative_url:
                url = response.urljoin(relative_url)
                yield scrapy.Request(url, callback=self.parse_detail)

        # Follow pagination using the correct HTML structure
        # Look for pagination container
        self.logger.info(f"Checking pagination on page: {response.url}")
        pagination_list = response.css("ul.number-list")
        
        if pagination_list:
            self.logger.info("Found pagination container")
            
            # Find current page (the one with class "red")
            current_page_link = pagination_list.css("a.page-number.red")
            current_page_text = current_page_link.css("::text").get()
            
            # Find all page links
            page_links = pagination_list.css("a.page-number")
            
            self.logger.info(f"Found {len(page_links)} page links")
            self.logger.info(f"Current page: {current_page_text}")
            
            # Look for the next page link by finding the next number after current page
            if current_page_text:
                try:
                    current_num = int(current_page_text.strip())
                    self.logger.info(f"Current page number: {current_num}")
                    
                    # Look for next page
                    for link in page_links:
                        page_num_text = link.css("::text").get()
                        if page_num_text and page_num_text.strip().isdigit():
                            page_num = int(page_num_text.strip())
                            self.logger.info(f"Found page link: {page_num}")
                            if page_num == current_num + 1:
                                next_url = link.css("::attr(href)").get()
                                if next_url:
                                    self.logger.info(f"Following next page: {next_url}")
                                    yield response.follow(next_url, self.parse)
                                    break
                    else:
                        self.logger.info("No next page found - reached end of pagination")
                        
                except ValueError:
                    self.logger.error(f"Could not parse current page number: {current_page_text}")
            else:
                # If no current page found, we might be on page 1, look for page 2
                self.logger.info("No current page with 'red' class found, looking for page 2")
                for link in page_links:
                    page_num_text = link.css("::text").get()
                    if page_num_text and page_num_text.strip() == "2":
                        next_url = link.css("::attr(href)").get()
                        if next_url:
                            self.logger.info(f"Following page 2: {next_url}")
                            yield response.follow(next_url, self.parse)
                            break
        else:
            self.logger.info("No pagination container found")

    def parse_detail(self, response):
        def extract_value(label):
            xpath = f'//li[span[contains(text(), "{label}")]]/a[@class="value-chars"]/text() | //li[span[contains(text(), "{label}")]]/span[@class="value-chars"]/text()'
            return response.xpath(xpath).get(default="").strip()

        def extract_int(text):
            numbers = re.findall(r'\d+', text)
            return int(numbers[0]) if numbers else None

        def extract_price():
            # Try to extract price from meta tag first
            price_meta = response.css('meta[itemprop="price"]::attr(content)').get()
            if price_meta:
                return float(price_meta)
            
            # Fallback: extract from announcement-price div
            price_text = response.css('div.announcement-price__cost::text').get()
            if price_text:
                price_numbers = re.findall(r'[\d\s]+', price_text.strip())
                if price_numbers:
                    return int(price_numbers[0].replace(' ', ''))
            return None

        area_str = extract_value("Площадь")
        floor_str = extract_value("Этаж")
        area = extract_int(area_str)
        floor = extract_int(floor_str)
        
        # Extract price
        price = extract_price()
        
        # Extract ad number
        ad_number = response.css('span[itemprop="sku"]::text').get()
        
        # Extract publication date
        pub_date = response.css('span.date-meta::text').get()
        if pub_date:
            pub_date = pub_date.replace('Опубликовано: ', '').strip()

        # Extract images
        image_elements = response.css('img.announcement__images-item.js-image-show-full')
        image_urls = image_elements.css('::attr(src)').getall()
        photo_count = len(image_urls)
        
        # Join image URLs with semicolon separator for CSV storage
        images_string = ';'.join(image_urls) if image_urls else ""

        yield {
            "url": response.url,
            "price": price,
            "ad_number": ad_number,
            "publication_date": pub_date,
            "area_m2": area,
            "floor": floor,
            "build_type": extract_value("Тип застройки"),
            "renovation": extract_value("Ремонт"),
            "bathroom": extract_value("Санузел"),
            "district": extract_value("Район"),
            "heating": extract_value("Отопление"),
            "built_status": extract_value("Состояние"),
            "tech_passport": extract_value("Техпаспорт"),
            "image_urls": images_string,
            "photo_count": photo_count,
        }

