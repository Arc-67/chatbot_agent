# crawler.py
import time
from datetime import datetime, timezone
import json
import logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE = "https://shop.zuscoffee.com"
COLLECTION = "https://shop.zuscoffee.com/collections/drinkware"

HEADERS = {
    "User-Agent": "ZUS-Drinkware-Crawler/1.0 (+https://yourdomain.example, contact: you@example.com)"
}
RATE_SEC = 0.6  # conservative rate

logging.basicConfig(level=logging.INFO)

def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_product_links(soup):
    # On the collection page product anchors often are <a href="/products/...">
    links = set()
    for a in soup.select("a[href*='/products/']"):
        href = a.get("href")
        if href:
            # normalise
            parsed = urljoin(BASE, href.split("?")[0])
            links.add(parsed)
    return sorted(links)

def scrape_product(url):
    soup = get_soup(url)
    # title
    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else None

    # price
    price_el = soup.select_one("[class*=price], .product-single__price")
    price = price_el.get_text(strip=True) if price_el else None

    # description (html and text)
    desc_block = soup.select_one("#ProductAccordion, .product-single__description, .product-form__description")
    if not desc_block:
        # fallback to any product description container
        desc_block = soup.select_one("[data-product-description]") or soup.find("div", {"class": "product-description"})
    description_html = str(desc_block) if desc_block else ""
    description_text = desc_block.get_text(" ", strip=True) if desc_block else ""

    # # images
    # images = []
    # for img in soup.select("img"):
    #     src = img.get("data-src") or img.get("src")
    #     if src and ("/products/" in src or "/files/" in src or "/cdn/" in src):
    #         images.append(urljoin(BASE, src.split("?")[0]))

    # product id or handle
    handle = url.rstrip("/").split("/")[-1]

    # variants (if present)
    variants = []
    for v in soup.select("select[name='id'] option"):
        variants.append({"id": v.get("value"), "label": v.get_text(strip=True)})

    return {
        "product_id": handle,
        "url": url,
        "title": title,
        "price": price,
        "description_text": description_text,
        "description_html": description_html,
        # "images": list(dict.fromkeys(images)),
        "variants": variants,
        "last_crawled": datetime.now(timezone.utc).isoformat(),
        "source": "zus_drinkware"
    }

def crawl_collection(collection_url, out_file="products.jsonl"):
    logging.info("Fetching collection page")
    soup = get_soup(collection_url)
    product_links = extract_product_links(soup)
    logging.info("Found %d product links", len(product_links))

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, p in enumerate(product_links, 1):
            try:
                logging.info("[%d/%d] Scraping %s", idx, len(product_links), p)
                product = scrape_product(p)
                f.write(json.dumps(product, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.exception("Failed to scrape %s: %s", p, e)
            time.sleep(RATE_SEC)

if __name__ == "__main__":
    crawl_collection(COLLECTION, out_file="products.jsonl")
