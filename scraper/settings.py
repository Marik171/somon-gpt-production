# Scrapy settings for somon_project project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'somon_project'

SPIDER_MODULES = ['scraper.spiders']
NEWSPIDER_MODULE = 'scraper.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# ===== OPTIMIZED FOR FAST SCRAPING =====
# Aggressive settings for near-real-time data collection
DOWNLOAD_DELAY = 0.1  # Very fast delay (100ms)
RANDOMIZE_DOWNLOAD_DELAY = 0.1  # Minimal randomization

# Increase concurrent requests for maximum speed
CONCURRENT_REQUESTS = 16  # Much more simultaneous requests
CONCURRENT_REQUESTS_PER_DOMAIN = 8  # Higher per domain concurrency

# AutoThrottle for adaptive speed control
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.1  # Start very fast
AUTOTHROTTLE_MAX_DELAY = 1.0    # Lower max delay
AUTOTHROTTLE_TARGET_CONCURRENCY = 5.0  # Higher target concurrent requests
AUTOTHROTTLE_DEBUG = False  # Set to True to see throttling stats

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# ===== ADDITIONAL SPEED OPTIMIZATIONS =====
# Disable unnecessary extensions for speed
TELNETCONSOLE_ENABLED = False

# Retry settings for reliability
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Timeout settings (faster timeouts)
DOWNLOAD_TIMEOUT = 8  # Shorter timeout for faster failure handling
RANDOMIZE_DOWNLOAD_DELAY = 0.1

# DNS settings for speed
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000

# Additional speed optimizations
COOKIES_ENABLED = False  # Disable cookies for speed
REDIRECT_ENABLED = True
REDIRECT_MAX_TIMES = 3

# Connection pool settings
DOWNLOAD_WARNSIZE = 33554432  # 32MB warning size
DOWNLOAD_MAXSIZE = 0  # No size limit for downloads

# Memory optimization for large scrapes
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 2048
MEMUSAGE_WARNING_MB = 1024

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'somon_project.middlewares.SomonProjectSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'somon_project.middlewares.SomonProjectDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'somon_project.pipelines.SomonProjectPipeline': 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
