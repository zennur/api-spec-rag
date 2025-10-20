import asyncio
from crawl4ai import *
from urllib.parse import urlparse

async def crawl_url(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        print(result.markdown)
        filename = extract_file_name(url)
        print(filename)
        

def extract_file_name(url)->str|None:
    host = urlparse(url).netloc.split('@')[-1].split(':')[0]  # strip creds/port
    labels = [l for l in host.split('.') if l]  # remove empty parts
    if not labels:
        return None
    # If there's a subdomain, usually provider is the second last label:
    # docs.stripe.com -> stripe, www.twilio.com -> twilio, api.openai.com -> openai
    if len(labels) >= 2:
        return labels[-2]
    # Single-label hosts (rare)
    return labels[0]

async def crawl_all_specs():

    for i in items:
        crawl_url(i['html_url']) 