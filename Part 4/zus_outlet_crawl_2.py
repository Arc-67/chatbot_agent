import httpx
from bs4 import BeautifulSoup
import json
import re
import asyncio

# The base URL for the category
BASE_URL = "https://zuscoffee.com/category/store/kuala-lumpur-selangor/"
OUTPUT_FILE = "outlets.jsonl"
TOTAL_PAGES = 22 # As you correctly identified

def clean_text(text: str) -> str:
    """Utility to clean whitespace, newlines, and unicode characters."""
    if not text:
        return "N/A"
    return (
        text.strip()
        .replace("\u2013", "-")  # Fix the en-dash '–'
        .replace("\u2019", "'")  # Fix smart quotes
        .replace("\u00a0", " ")  # Fix non-breaking spaces
        .replace("\ufffd", " ")  # Fix non-breaking spaces
        .replace("Operating Hours:", "")
        .replace("Address:", "")
    )

def extract_services(element) -> list:
    """Extracts services from the HTML, looking for keywords."""
    services = ["ZUS App Pickup"]  # Default service
    text = element.get_text(separator=" ").lower()
    
    if "dine-in" in text:
        services.append("Dine-in")
    if "24 hours" in text or "24-hour" in text:
        services.append("24 Hours")
    # Add other services as needed
    return list(set(services)) # Use set to remove duplicates

def parse_city_state(address: str) -> (str, str):
    """Parses city and state from an address string."""
    city, state = "N/A", "N/A"
    if address != "N/A":
        # Split by comma
        address_parts = [part.strip().strip(".") for part in address.split(",") if part.strip()]
        if len(address_parts) >= 2:
            # Heuristic: State is last part
            state = address_parts[-1]
            # City is second-to-last, clean postal code
            city_part = address_parts[-2]
            city = re.sub(r"^\d{5,}\s*", "", city_part).strip() # Remove postal code
        elif len(address_parts) == 1:
            state = address_parts[0] # Better than nothing
    
    # Handle common state misspellings or full names
    if "kuala lumpur" in state.lower():
        state = "Kuala Lumpur"
    if "selangor" in state.lower():
        state = "Selangor"

    return city, state

async def scrape_outlets():
    print(f"Starting advanced scrape. This will take ~30 seconds...")
    
    all_outlets = []
    
    async with httpx.AsyncClient(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }, follow_redirects=True) as client:

        # --- Scrape all 22 pages and parse data directly ---
        for page in range(1, TOTAL_PAGES + 1):
            current_url = BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"
            print(f"Scraping listing page {page}/{TOTAL_PAGES}...")
            
            try:
                response = await client.get(current_url, timeout=15.0)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find all store cards on the page
                store_cards = soup.find_all("article", class_=re.compile(r"elementor-post"))
                
                if not store_cards and page == 1:
                    print("Error: No store cards found on first page. Selectors may be broken.")
                    return

                for card in store_cards:
                    # --- Use the reliable selectors from your first debug log ---
                    
                    # 1. Find the Name
                    name_tag = card.find("p", class_="elementor-heading-title")
                    
                    # 2. Find the Address
                    content_div = card.find("div", {"data-widget_type": "theme-post-content.default"})
                    address_tag = content_div.find("p") if content_div else None
                    
                    # 3. Find the Google Maps link (as a sanity check)
                    maps_link = card.find("a", href=re.compile(r"maps\.app\.goo\.gl"))

                    # This is a real outlet card if it has a name, address, and maps link
                    if not (name_tag and address_tag and maps_link):
                        continue # Skip this <article> as it's not an outlet
                    
                    name = clean_text(name_tag.get_text(separator=" "))
                    address = clean_text(address_tag.get_text(separator=" "))
                    operating_hours = "N/A" # Not available on this page
                    services = extract_services(card)
                    city, state = parse_city_state(address)
                    
                    outlet_data = {
                        "name": name,
                        "address": address,
                        "city": city,
                        "state": state,
                        "operating_hours": operating_hours,
                        "services": services
                    }
                    all_outlets.append(outlet_data)
                    
            except httpx.RequestError as e:
                print(f"Error: Failed to fetch page {page}: {e}. Stopping.")
                break
            except Exception as e:
                print(f"Warning: Error parsing page {page}: {e}")
        
        print(f"\nFound {len(all_outlets)} total outlets across {TOTAL_PAGES} pages.")
        
        # --- Save all results to file at the end ---
        outlets_found = 0
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for data in all_outlets:
                if data: # Filter out None results
                    f.write(json.dumps(data) + "\n")
                    outlets_found += 1

        print(f"\nScraping complete. Successfully processed and saved {outlets_found} outlets to {OUTPUT_FILE}.")

if __name__ == "__main__":
    asyncio.run(scrape_outlets())