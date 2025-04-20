import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext
import fitz  # PyMuPDF
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import requests
import json
import time
import base64
import threading
import concurrent.futures
import io
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from collections import defaultdict
import hashlib
import re
import logging
from logging.handlers import RotatingFileHandler

# Common stop words to skip for AI categorization
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "or", "but", "if", "this", "they", "their", "them", "then", "when",
    "where", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "do", "does", "did", "doing"
}

# Add Pydantic imports at the top of the script
from pydantic import BaseModel, Field
from typing import List, Optional

# Define Pydantic models for API responses
class FlaggedItem(BaseModel):
    term: Optional[str] = None  # Make term optional with a default of None
    category: str
    severity: int = Field(ge=1, le=5)  # Severity between 1 and 5
    context: str

class APIResponse(BaseModel):
    choices: List[dict] = Field(default_factory=list)

# Configure logging with rotation
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a RotatingFileHandler
rotating_handler = RotatingFileHandler(
    filename='pdfscan.log',  # Updated log file name
    maxBytes=1_000_000,      # Rotate when file reaches 1 MB (1,000,000 bytes)
    backupCount=5            # Keep up to 5 backup files (pdascan.log.1, pdascan.log.2, ..., pdascan.log.5)
)

# Define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
rotating_handler.setFormatter(formatter)

# Add handlers to the logger
logger.handlers = []  # Clear any existing handlers
logger.addHandler(rotating_handler)  # Add the rotating file handler
logger.addHandler(logging.StreamHandler())  # Also output to console

BASE_URL = "https://api.x.ai/v1"
XAI_API_KEY = "xai-wXiK6UcD68YZWcayjPdGjhkZtrjR6YZUrna53SkdbaOWvQymKTjnt8H2y8xSOa2d4tBpsNDTGsnfUYfY"
output_dir = os.getcwd()

stop_scanning = False
BATCH_SIZE = 10
CHUNK_SIZE = 10
api_cache = {}

def grok_scan_content(content, is_image=False, retries=3, base_delay=5, challenge_report=False):
    if isinstance(content, io.BytesIO):
        content.seek(0)
        cache_key = hashlib.md5(content.read()).hexdigest()
        content.seek(0)
    else:
        cache_key = hashlib.md5(content.encode('utf-8')).hexdigest()
    cache_key += "_image" if is_image else "_text"
    cache_key += "_challenge" if challenge_report else "_standard"

    if cache_key in api_cache:
        print(f"Using cached API result for content hash: {cache_key}")
        logging.debug(f"Using cached API result for content hash: {cache_key}")
        return api_cache[cache_key], True

    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    
    if is_image:
        if isinstance(content, io.BytesIO):
            content.seek(0)
            try:
                img = Image.open(content)
                img.verify()
                content.seek(0)
                if content.getbuffer().nbytes == 0:
                    print("Empty image stream detected, skipping API call")
                    logging.debug("Empty image stream detected, skipping API call")
                    return [], False
                img_base64 = base64.b64encode(content.read()).decode("utf-8")
            except Exception as e:
                print(f"Invalid or empty image data: {e}")
                logging.error(f"Invalid or empty image data: {e}")
                return [], False
        else:
            with open(content, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        if not img_base64:
            print("Failed to encode image to base64, skipping API call")
            logging.debug("Failed to encode image to base64, skipping API call")
            return [], False

        if challenge_report:
            prompt = (
                "You are an educational content analyst. Analyze the provided image for appropriateness in a school setting, considering: "
                "1. Nudity (e.g., explicit vs. artistic/educational), "
                "2. Violence (e.g., graphic vs. historical depiction), "
                "3. LGBTQ-related symbols (e.g., advocacy vs. neutral), "
                "4. Anti-government/religious/American imagery (e.g., satirical vs. informational), "
                "5. Egregious or abhorrent content (e.g., racial slurs, divisive rhetoric), "
                "6. DEI (e.g., diversity, equity, inclusion), "
                "7. Critical Race Theory (e.g., white privilege), "
                "8. Critical Gender Theory (e.g., gender theory), "
                "9. Marxist indoctrination (e.g., Marxism). "
                "Return a JSON list of objects with 'term', 'category', 'severity' (1-5, 1=mild/contextually appropriate, 5=severe/inappropriate), and 'context'. "
                "The 'term' field should be a specific word or phrase that triggered the flag (e.g., 'nudity', 'violence'). "
                "Example: [{'term': 'nudity', 'category': 'Nudity', 'severity': 4, 'context': 'explicit depiction'}]."
            )
        else:
            prompt = (
                "You are a content moderator. Analyze the provided image and identify if it contains: "
                "1. Nudity (e.g., exposed private parts, explicit poses), "
                "2. Violence (e.g., blood, gore, physical harm), "
                "3. LGBTQ-related symbols (e.g., pride flags, trans flags), "
                "4. Symbols or imagery that could be perceived as anti-government, anti-religious, or anti-American "
                "(e.g., defaced flags, mocking religious icons), "
                "5. Egregious or abhorrent content (e.g., racial slurs, divisive rhetoric), "
                "6. Content related to DEI (Diversity, Equity, & Inclusion) (e.g., diversity, equity, inclusion, systemic racism), "
                "7. Content related to Critical Race Theory (e.g., critical race, white privilege, structural racism), "
                "8. Content related to Critical Gender Theory (e.g., gender theory, non-binary, transgender ideology), "
                "9. Content related to Marxist indoctrination (e.g., Marxism, class struggle, bourgeoisie, proletariat). "
                "Return only a JSON list of flagged categories or terms (e.g., ['nudity', 'racism']) "
                "or an empty list [] if none are found. Do not include any additional text or explanation."
            )

        payload = {
            "model": "grok-2-vision-latest",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}
            ],
            "max_tokens": 1000,  # Increased from 500 to handle larger responses
            "temperature": 0.5
        }
    else:
        if not content or not content.strip():
            print("Empty text content detected, skipping API call")
            logging.debug("Empty text content detected, skipping API call")
            return [], False

        if challenge_report:
            prompt = (
                "You are an educational content analyst evaluating text for appropriateness in a school setting. Analyze the provided text and identify terms, phrases, or themes that could be inappropriate or sensitive, considering: "
                "1. Sexual content (e.g., explicit descriptions, casual sex references, distinguish educational vs. provocative intent), "
                "2. Violence (e.g., graphic depictions vs. historical references), "
                "3. Drug use (e.g., recreational vs. medical context), "
                "4. Foul language (e.g., casual profanity vs. quoted dialogue), "
                "5. Sensitive social topics (e.g., LGBTQ issues, distinguish advocacy vs. neutral discussion), "
                "6. Political/ideological content (e.g., anti-government rhetoric, assess tone and intent), "
                "7. Educational inappropriateness (e.g., content offensive to conservative values or too mature for young readers), "
                "8. Egregious content (e.g., slurs, hate speech, grooming implications), "
                "9. DEI (e.g., diversity, equity, inclusion), "
                "10. Critical Race Theory (e.g., white privilege), "
                "11. Critical Gender Theory (e.g., gender theory), "
                "12. Marxist indoctrination (e.g., Marxism), "
                "13. Alcohol use (e.g., beer). "
                "Return a JSON list of objects with 'term', 'category', 'severity' (1-5, 1=mild/contextually appropriate, 5=severe/inappropriate), and 'context'. "
                "Example: [{'term': 'sex', 'category': 'Sexual Content', 'severity': 2, 'context': 'biological education'}]."
            )
        else:
            prompt = (
                "You are a content moderator tasked with identifying specific inappropriate or sensitive content. "
                "Scan the provided text and identify terms or phrases related to: "
                "1. Sexual content (e.g., intercourse, explicit descriptions, sex), "
                "2. Violence (e.g., murder, assault, gore, gun), "
                "3. Drug use (e.g., cocaine, marijuana, addiction, meth), "
                "4. Foul language (e.g., profanity, slurs), "
                "5. Sensitive social topics (e.g., LGBTQ, transgender issues, suicide, self-harm, gay), "
                "6. Political or ideological content (e.g., anti-government sentiments like 'government corruption,' "
                "anti-religious sentiments like 'religion is a scam,' anti-American sentiments like 'America is a failed state'), "
                "7. Content that may be considered offensive or inappropriate in an educational setting "
                "(e.g., explicit sexual education, extremist views, material offensive to conservative parents such as alternative lifestyle discussions), "
                "8. Egregious or abhorrent content (e.g., racial slurs, divisive rhetoric), "
                "9. Content related to DEI (Diversity, Equity, & Inclusion) (e.g., diversity, equity, inclusion, systemic racism), "
                "10. Content related to Critical Race Theory (e.g., critical race, white privilege, structural racism), "
                "11. Content related to Critical Gender Theory (e.g., gender theory, non-binary, transgender ideology), "
                "12. Content related to Marxist indoctrination (e.g., Marxism, class struggle, bourgeoisie, proletariat), "
                "13. Alcohol use (e.g., beer, drinking alcohol). "
                "Return only a JSON list of the exact terms or phrases found (e.g., ['violence', 'systemic racism']) "
                "or an empty list [] if none are detected. Do not include any additional text or explanation."
            )

        payload = {
            "model": "grok-2-latest",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1000,  # Increased from 500 to handle larger responses
            "temperature": 0.5
        }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 429:  # Rate limit exceeded
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{retries})")
                logging.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                continue
            response.raise_for_status()
            result = response.json()
            # Validate API response using Pydantic
            api_response = APIResponse(**result)
            flagged_content = api_response.choices[0].get("message", {}).get("content", "[]")
            flagged_content = flagged_content.strip()

            # Check for incomplete JSON (e.g., missing closing brackets)
            if flagged_content.count('[') != flagged_content.count(']') or flagged_content.count('{') != flagged_content.count('}'):
                if attempt < retries - 1:
                    payload["max_tokens"] += 500  # Increase tokens and retry
                    print(f"Incomplete JSON detected, increasing max_tokens to {payload['max_tokens']} and retrying (attempt {attempt + 1}/{retries})")
                    logging.warning(f"Incomplete JSON detected, increasing max_tokens to {payload['max_tokens']} and retrying (attempt {attempt + 1}/{retries})")
                    if isinstance(content, io.BytesIO):
                        content.seek(0)
                    continue
                else:
                    print(f"Failed to get complete JSON after {retries} attempts: {flagged_content}")
                    logging.error(f"Failed to get complete JSON after {retries} attempts: {flagged_content}")
                    return [], False

            # Extract JSON from potential markdown
            json_start = flagged_content.find("```json")
            json_end = flagged_content.rfind("```")
            if json_start != -1 and json_end != -1 and json_start < json_end:
                flagged_content = flagged_content[json_start + 7:json_end].strip()
            elif flagged_content.startswith("```json") and flagged_content.endswith("```"):
                flagged_content = flagged_content[7:-3].strip()
            elif flagged_content.startswith("```") and flagged_content.endswith("```"):
                flagged_content = flagged_content[3:-3].strip()
            elif not flagged_content:  # Handle empty response
                print("Empty API response received, treating as empty list")
                logging.warning("Empty API response received, treating as empty list")
                return [], False

            try:
                parsed = json.loads(flagged_content)
                if not isinstance(parsed, list):
                    print(f"Warning: API returned non-list response: {parsed}")
                    logging.warning(f"API returned non-list response: {parsed}")
                    return [], False
                if challenge_report:
                    # Validate and infer term if missing
                    validated_items = []
                    for item in parsed:
                        if isinstance(item, dict):
                            if 'term' not in item or item['term'] is None:
                                item['term'] = item.get('category', 'unknown').lower()
                            validated_items.append(FlaggedItem(**item).model_dump())
                        else:
                            print(f"Skipping invalid item: {item}")
                            logging.warning(f"Skipping invalid item: {item}")
                    parsed = validated_items
                else:
                    parsed = [item.get('term', item.get('category', 'unknown')) if isinstance(item, dict) else item for item in parsed]
                api_cache[cache_key] = parsed
                return parsed, False
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} - Raw response: {flagged_content}")
                logging.error(f"JSON decode error: {e} - Raw response: {flagged_content}")
                return [], False
            except Exception as e:
                print(f"Validation error: {e} - Raw response: {flagged_content}")
                logging.error(f"Validation error: {e} - Raw response: {flagged_content}")
                return [], False
        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
            logging.error(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(base_delay)
                if isinstance(content, io.BytesIO):
                    content.seek(0)
                continue
            return [], False
    return [], False
			
def flatten_terms(terms):
    flat_list = []
    if isinstance(terms, (list, tuple)):
        for item in terms:
            if isinstance(item, (list, tuple)):
                flat_list.extend(flatten_terms(item))
            elif isinstance(item, dict):
                flat_list.append(item.get('term', item.get('category', 'unknown')))
            elif isinstance(item, str):
                flat_list.append(item)
    elif isinstance(terms, str):
        flat_list.append(terms)
    return flat_list

def generate_grok_report(flagged_content, title, author, flagged_page_width, flagged_page_height):
    if not flagged_content:
        return None
    
    terms_for_report = flatten_terms(flagged_content)
    prompt = (
        "You are an educational content analyst. Based on the following flagged content from a book, "
        "generate a brief report (150-200 words) including: "
        "1. A summary of the flagged content found in the book (e.g., types of terms or imagery). "
        "2. Potential harm this content could have on young, impressionable students (e.g., emotional distress, inappropriate exposure). "
        "3. A recommended minimum age requirement for exposure to this content, with a brief justification. Take into consideration that the age of sexual consent in most US states is 18 years old. A child should NEVER keep secrets from their parents - especially if an adult asks you to. No use of the internet to meet strangers is ever permitted for someone under the age of 18."
        "Format the response as plain text, suitable for inclusion in a PDF report page. "
        "Use ** to denote section headers (e.g., **Summary of Flagged Content**). "
        f"Book Title: {title or 'Unknown'}\n"
        f"Author: {author or 'Unknown'}\n"
        f"Flagged Content: {', '.join(sorted(set(terms_for_report)))}"
    )
    
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-2-latest",
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        report_text = result.get("choices", [{}])[0].get("message", {}).get("content", "Report generation failed.")
        print(f"Full report text received: {report_text}")
        
        if report_text.endswith("which could") or report_text.endswith("which could "):
            print("Warning: Report appears truncated mid-sentence. Retrying with higher max_tokens.")
            payload["max_tokens"] = 700
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            report_text = result.get("choices", [{}])[0].get("message", {}).get("content", report_text)
            print(f"Retried with higher tokens, new text: {report_text}")

        report_img = Image.new("RGB", (flagged_page_width, flagged_page_height), "white")
        draw = ImageDraw.Draw(report_img)

        scale_width = flagged_page_width / 800
        scale_height = flagged_page_height / 800
        try:
            title_font = ImageFont.truetype("arial.ttf", int(40 * min(scale_width, scale_height)))
        except Exception:
            title_font = ImageFont.load_default().font_variant(size=int(40 * min(scale_width, scale_height)))
        try:
            body_font = ImageFont.truetype("arial.ttf", int(16 * min(scale_width, scale_height)))
        except Exception:
            body_font = ImageFont.load_default().font_variant(size=int(16 * min(scale_width, scale_height)))
        try:
            bold_font = ImageFont.truetype("arialbd.ttf", int(16 * min(scale_width, scale_height)))
        except Exception:
            bold_font = ImageFont.load_default().font_variant(size=int(16 * min(scale_width, scale_height)))

        title_text = "Content Analysis Report"
        title_width = draw.textlength(title_text, font=title_font)
        draw.text(((flagged_page_width - title_width) / 2, int(50 * scale_height)), title_text, font=title_font, fill="black")

        y_offset = int(100 * scale_height)
        margin_left = int(50 * scale_width)
        max_width = flagged_page_width - 2 * margin_left
        line_spacing = int(20 * min(scale_width, scale_height))
        section_spacing = int(30 * min(scale_width, scale_height))

        sections = re.split(r'\*\*(.*?)\*\*', report_text)
        current_font = body_font
        extra_spacing = 0

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            if i % 2 == 1:
                current_font = bold_font
                y_offset += section_spacing
            else:
                current_font = body_font
                section_spacing = 0

            lines = []
            current_line = ""
            for word in section.split():
                test_line = f"{current_line} {word}" if current_line else word
                if draw.textlength(test_line, font=current_font) <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            for j, line in enumerate(lines):
                if y_offset > flagged_page_height - int(50 * scale_height):
                    print(f"Warning: Report truncated at line {j} of section {i} due to page height limit. Remaining text: {lines[j:]}")
                    break
                draw.text((margin_left, y_offset), line, font=current_font, fill="black")
                y_offset += line_spacing
            else:
                print(f"Section {i} fully rendered, y_offset: {y_offset}")

        print(f"Final y_offset: {y_offset}, Page height limit: {flagged_page_height - int(50 * scale_height)}")
        return report_img
    
    except Exception as e:
        print(f"Failed to generate report: {e}")
        logging.error(f"Failed to generate report: {e}")
        return None

def generate_index_page(title, author, cover_image, category_tally, flagged_page_width, flagged_page_height, output_dir, sanitized_title, temp_image_paths, challenge_report=False):
    # List to store paths of all generated index pages
    index_paths = []

    # Prepare fonts and scaling
    scale_width = flagged_page_width / 800
    scale_height = flagged_page_height / 800

    try:
        title_font = ImageFont.truetype("arial.ttf", int(40 * min(scale_width, scale_height)))
    except Exception:
        title_font = ImageFont.load_default().font_variant(size=int(40 * min(scale_width, scale_height)))
    try:
        category_font = ImageFont.truetype("arialbd.ttf", int(12 * min(scale_width, scale_height)))  # Bold Arial
    except Exception:
        category_font = ImageFont.load_default().font_variant(size=int(12 * min(scale_width, scale_height)))
    try:
        tally_font = ImageFont.truetype("arial.ttf", int(12 * min(scale_width, scale_height)))
    except Exception:
        tally_font = ImageFont.load_default().font_variant(size=int(12 * min(scale_width, scale_height)))

    # Create a temporary image for text length calculations
    temp_img = Image.new("RGB", (flagged_page_width, flagged_page_height), "white")
    draw = ImageDraw.Draw(temp_img)

    # Prepare items for display
    all_items = []
    for category in sorted(category_tally.keys()):
        category_with_colon = f"{category}:"
        wrapped_category = [category_with_colon]
        col_width = int(flagged_page_width / 4 - 20 * scale_width)
        if draw.textlength(category_with_colon, font=category_font) > col_width:
            words = category.split()
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}" if current_line else word
                if draw.textlength(test_line + ":", font=category_font) <= col_width:
                    current_line = test_line
                else:
                    if current_line:
                        wrapped_category.append(current_line + ":")
                    current_line = word
            if current_line:
                wrapped_category.append(current_line + ":")
        for line in wrapped_category:
            all_items.append(("category", line))
        terms = category_tally[category]
        sorted_terms = sorted(terms.items(), key=lambda x: x[0].lower())
        for i, (term, count) in enumerate(sorted_terms, 1):
            if count > 0:
                all_items.append(("term", f"{i}. {term}: {count}"))

    # If no tally, add a placeholder
    if not all_items:
        all_items.append(("category", "Offensive Content Tally:"))
        all_items.append(("term", "No Matches Found in Predefined Egregious Words Dictionary"))

    # Split items across multiple pages
    item_height = int(14 * scale_height)
    col_width = int(flagged_page_width / 4 - 20 * scale_width)
    margin_bottom = int(50 * scale_height)
    num_columns = 4
    page_height_limit = flagged_page_height - margin_bottom
    page_number = 1
    item_idx = 0

    while True:
        # Create a new index page
        index_img = Image.new("RGB", (flagged_page_width, flagged_page_height), "white")
        draw = ImageDraw.Draw(index_img)

        # Draw title and author (only on first page)
        if page_number == 1:
            title_text = f"Title: {title or 'Unknown'}"
            author_text = f"Author: {author or 'Unknown'}"
            title_width = draw.textlength(title_text, font=title_font)
            author_width = draw.textlength(author_text, font=title_font)
            draw.text(((flagged_page_width - title_width) / 2, int(10 * scale_height)), title_text, font=title_font, fill="black")
            draw.text(((flagged_page_width - author_width) / 2, int(60 * scale_height)), author_text, font=title_font, fill="black")

            # Draw cover image if available
            cover_y_offset = 0
            cover_start_y = int(110 * scale_height)
            if cover_image:
                orig_width, orig_height = cover_image.size
                target_width = int(200 * scale_width)
                target_height = int(target_width * orig_height / orig_width)
                if target_height > int(300 * scale_height):
                    target_height = int(300 * scale_height)
                    target_width = int(target_height * orig_width / orig_height)
                cover = cover_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                border_img = Image.new("RGB", (target_width + int(10 * scale_width), target_height + int(10 * scale_height)), "black")
                border_img.paste(cover, (int(5 * scale_width), int(5 * scale_height)))
                index_img.paste(border_img, (int(10 * scale_width), cover_start_y))
                cover_y_offset = target_height + int(10 * scale_height)

            # Draw robot image if available
            robot_img_path = "robot_input.png"
            robot_y_offset = 0
            if os.path.exists(robot_img_path):
                robot_img = Image.open(robot_img_path).convert("RGBA")
                orig_width, orig_height = robot_img.size
                target_height = int(300 * scale_height)
                aspect_ratio = orig_width / orig_height
                target_width = int(target_height * aspect_ratio)
                if target_width > int(300 * scale_width):
                    target_width = int(300 * scale_width)
                    target_height = int(target_width / aspect_ratio)
                robot_img = robot_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                robot_border_img = Image.new("RGBA", (int(target_width + 10 * scale_width), int(target_height + 10 * scale_height)), (0, 0, 0, 0))
                robot_border_img.paste(robot_img, (int(5 * scale_width), int(5 * scale_height)))
                robot_border_rgb = Image.new("RGB", robot_border_img.size, "white")
                robot_border_rgb.paste(robot_border_img, mask=robot_border_img.split()[3])
                index_img.paste(robot_border_rgb, (int(flagged_page_width - (target_width + 10 * scale_width) - 10 * scale_width), cover_start_y))
                robot_y_offset = target_height + int(10 * scale_height)

            # Calculate starting y_offset for tally section
            max_image_height = max(cover_y_offset, robot_y_offset)
            y_offset = cover_start_y + max_image_height + int(20 * scale_height)
        else:
            # On subsequent pages, start tally higher up
            y_offset = int(20 * scale_height)

        # Draw tally header
        header_text = f"Offensive Content Tally (Page {page_number}):"
        draw.text((int(10 * scale_width), y_offset), header_text, font=category_font, fill="black")
        y_offset += int(14 * scale_height)

        # Display items on the current page
        column_y_positions = [y_offset for _ in range(num_columns)]
        start_idx = item_idx
        while item_idx < len(all_items):
            min_y_idx = column_y_positions.index(min(column_y_positions))
            x_offset = int(10 * scale_width + min_y_idx * (col_width + 10 * scale_width))
            y = column_y_positions[min_y_idx]

            if y + item_height > page_height_limit:
                break  # Move to next page

            item_type, item = all_items[item_idx]
            if item_type == "category":
                draw.text((x_offset, y), item, font=category_font, fill="black")
            else:
                draw.text((x_offset, y), item, font=tally_font, fill="black")
            column_y_positions[min_y_idx] += item_height
            item_idx += 1

        # Save the current index page
        index_path = os.path.join(output_dir, f"{sanitized_title}_index_page_{page_number}.png")
        index_img.save(index_path)
        temp_image_paths.add(index_path)
        index_paths.append(index_path)
        logging.debug(f"Index page {page_number} saved at: {index_path}")

        # If all items are displayed, break
        if item_idx >= len(all_items):
            break

        page_number += 1

    return index_paths

def process_pdf(pdf_path, title, author, cover_image_path, output_dir, app):
    global stop_scanning
    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        output_images = []
        temp_image_paths = set()
        all_flagged_content = []
        page_flagged_content = {}
        pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        cover_image = None
        if not cover_image_path or not os.path.exists(cover_image_path):
            pix = pdf_doc[0].get_pixmap(dpi=200)
            if pix.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            cover_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            cover_image = Image.open(cover_image_path)

        text_pages = {}
        image_tasks = []

        # Use Turbo Mode if enabled (8 threads), else default to 4
        max_workers = 8 if app.turbo_mode_var.get() else 4
        logging.debug(f"Using {max_workers} threads for processing (Turbo Mode: {app.turbo_mode_var.get()})")
        app.status_label.config(text=f"Loading document with {max_workers} threads...")

        def extract_page_data(page_num):
            if stop_scanning:
                return None, None, None
            page = pdf_doc[page_num]
            text = page.get_text("text") or ""
            images = page.get_images(full=True)
            page_images = []
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    temp_img = io.BytesIO()
                    image = image.resize((int(image.width * 0.9), int(image.height * 0.9)), Image.Resampling.LANCZOS)
                    image.save(temp_img, format="PNG", optimize=True, quality=95)
                    temp_img.seek(0)
                    page_images.append((temp_img, page_num + 1))
                    print(f"Extracted image {img_index + 1} for page {page_num + 1}")
                    logging.debug(f"Extracted image {img_index + 1} for page {page_num + 1}")
                except Exception as e:
                    print(f"Failed to extract image {img_index + 1} for page {page_num + 1}: {e}")
                    logging.error(f"Image extraction failed for page {page_num + 1}: {e}")
            return page_num + 1, text, page_images

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            logging.debug(f"Active threads before page extraction: {threading.active_count()}")
            futures = [executor.submit(extract_page_data, page_num) for page_num in range(total_pages)]
            for future in concurrent.futures.as_completed(futures):
                if stop_scanning:
                    app.update_progress(0, total_pages, "Scanning stopped by user.", None, None)
                    break
                result = future.result()
                if result:
                    page_num, text, page_images = result
                    app.update_progress(page_num - 1, total_pages, f"Extracting text and images for page {page_num} of {total_pages}...", None, None)
                    text_pages[page_num] = text
                    image_tasks.extend(page_images)
                else:
                    logging.warning(f"Page {page_num} extraction returned None")
            logging.debug(f"Active threads after page extraction: {threading.active_count()}")
        logging.debug(f"Page extraction took {time.time() - start_time:.2f} seconds with {max_workers} threads")

        saved_pages = set()
        for chunk_start in range(0, total_pages, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
            app.update_progress(chunk_start, total_pages, f"Processing pages {chunk_start + 1} to {chunk_end} of {total_pages}...", None, None)

            text_results = {}
            for page_num in range(chunk_start + 1, chunk_end + 1):
                if stop_scanning:
                    app.update_progress(page_num - 1, total_pages, "Scanning stopped by user.", None, None)
                    break
                text = text_pages.get(page_num, "")
                results, used_cache = grok_scan_content(text, is_image=False, challenge_report=app.challenge_report_var.get())
                text_results[page_num] = results
                cache_msg = " (cached)" if used_cache else ""
                print(f"Page {page_num} text flagged_content: {results}")
                logging.debug(f"Page {page_num} text flagged_content: {results}")
                app.update_progress(page_num - 1, total_pages, f"Scanning text for page {page_num}{cache_msg}...", None, results)

            image_results = defaultdict(list)
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                logging.debug(f"Active threads before image scanning: {threading.active_count()}")
                future_to_task = {executor.submit(grok_scan_content, task[0], is_image=True, challenge_report=app.challenge_report_var.get()): task for task in image_tasks if task[1] in range(chunk_start + 1, chunk_end + 1)}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    page_num = task[1]
                    try:
                        result, used_cache = future.result()
                        image_results[page_num].append(result)
                        cache_msg = " (cached)" if used_cache else ""
                        app.update_progress(page_num - 1, total_pages, f"Scanning image for page {page_num}{cache_msg}...", task[0], result)
                        print(f"Page {page_num} image flagged_content: {result}")
                        logging.debug(f"Page {page_num} image flagged_content: {result}")
                    except Exception as e:
                        print(f"Image scan failed for page {page_num}: {e}")
                        logging.error(f"Image scan failed for page {page_num}: {e}")
                        image_results[page_num].append([])
                logging.debug(f"Active threads after image scanning: {threading.active_count()}")
            logging.debug(f"Image scanning took {time.time() - start_time:.2f} seconds with {max_workers} threads")

            for page_num in range(chunk_start, chunk_end):
                if stop_scanning:
                    app.update_progress(page_num, total_pages, "Scanning stopped by user.", None, None)
                    break
                page_key = page_num + 1
                text_flags = text_results.get(page_key, [])
                image_flags = [item for sublist in image_results.get(page_key, []) for item in sublist]
                flagged_content = text_flags + image_flags
                page_flagged_content[page_key] = flagged_content
                app.update_progress(page_num, total_pages, f"Processing page {page_key} of {total_pages}...", None, flagged_content)

                if flagged_content and page_key not in saved_pages:
                    try:
                        page = pdf_doc[page_num]
                        for term in flatten_terms(text_flags):
                            try:
                                for rect in page.search_for(term):
                                    highlight = page.add_highlight_annot(rect)
                                    highlight.set_colors((1, 0, 0, 0.5))
                                    highlight.update()
                            except Exception as e:
                                logging.error(f"Highlighting failed for term '{term}' on page {page_key}: {e}")
                        pix = page.get_pixmap(dpi=200)
                        if pix.n > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        if pix.width == 0 or pix.height == 0 or not pix.samples:
                            print(f"Invalid pixmap for page {page_key}")
                            logging.warning(f"Invalid pixmap for page {page_key}")
                            continue
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.Resampling.LANCZOS)
                        img_path = os.path.join(output_dir, f"{pdf_base_name}_page_{page_key}.png")
                        img.save(img_path, optimize=True, quality=95)
                        output_images.append(img_path)
                        temp_image_paths.add(img_path)
                        saved_pages.add(page_key)
                        all_flagged_content.extend(flagged_content)
                    except Exception as e:
                        print(f"Error saving page {page_key}: {str(e)}")
                        logging.error(f"Error saving page {page_key}: {e}")

        # ... (Rest of the function unchanged: sorting, dimensions, return statement) ...
        if app.challenge_report_var.get():
            output_images_with_pages = []
            for img_path in output_images:
                page_num_str = img_path.split('_page_')[-1].replace('.png', '')
                page_num = int(page_num_str)
                output_images_with_pages.append((page_num, img_path))
            output_images_with_pages.sort(key=lambda x: x[0])
            output_images = [path for _, path in output_images_with_pages]
            sorted_page_flagged_content = {}
            for page_num, _ in output_images_with_pages:
                sorted_page_flagged_content[page_num] = page_flagged_content.get(page_num, [])
            page_flagged_content = sorted_page_flagged_content

        flagged_page_width, flagged_page_height = None, None
        if output_images:
            first_flagged_img = Image.open(output_images[0])
            flagged_page_width, flagged_page_height = first_flagged_img.size
            print(f"Flagged page dimensions (from saved image): {flagged_page_width}x{flagged_page_height}")
            logging.debug(f"Flagged page dimensions (from saved image): {flagged_page_width}x{flagged_page_height}")
        else:
            flagged_page_width, flagged_page_height = 612, 792
            print("No flagged images generated, using fallback dimensions: 612x792")
            logging.debug("No flagged images generated, using fallback dimensions: 612x792")

        return pdf_doc, cover_image, total_pages, output_images, all_flagged_content, None, pdf_base_name, image_tasks, flagged_page_width, flagged_page_height, None, text_pages, page_flagged_content, temp_image_paths

    except Exception as e:
        app.update_progress(0, 0, f"Error: {str(e)}", None, None)
        logging.error(f"Error in process_pdf: {e}")
        return None, None, 0, None, None, None, None, None, None, None, f"Error processing PDF: {str(e)}", {}, {}, set()
		
def process_epub(epub_path, title, author, cover_image_path, output_dir, app):
    global stop_scanning
    try:
        book = epub.read_epub(epub_path)
        total_pages = 0
        output_images = []
        temp_image_paths = set()
        all_flagged_content = []
        page_flagged_content = {}
        epub_base_name = os.path.splitext(os.path.basename(epub_path))[0]

        cover_image = None
        if cover_image_path and os.path.exists(cover_image_path):
            cover_image = Image.open(cover_image_path)
        else:
            for item in book.get_items_of_type(ebooklib.ITEM_COVER):
                cover_data = item.get_content()
                cover_image = Image.open(io.BytesIO(cover_data))
                break
            if not cover_image:
                cover_image = Image.new("RGB", (200, 300), "gray")

        text_pages = {}
        image_tasks = []
        page_num = 0

        html_items = [item for item in book.get_items() if item.get_name().lower().endswith(('.html', '.xhtml')) and 'html' in item.get_name().lower()]
        html_items.sort(key=lambda x: x.get_name())

        # Use Turbo Mode if enabled (8 threads), else default to 4
        max_workers = 8 if app.turbo_mode_var.get() else 4

        def extract_epub_page(item, current_page):
            if stop_scanning:
                return None, None, None
            content = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text() or ""
            page_images = []
            print(f"Extracting images for EPUB page {current_page}: found {len(soup.find_all('img'))} images")
            logging.debug(f"Extracting images for EPUB page {current_page}: found {len(soup.find_all('img'))} images")
            for img in soup.find_all("img"):
                src = img.get("src")
                for epub_item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                    if src in epub_item.get_name():
                        img_data = epub_item.get_content()
                        image = Image.open(io.BytesIO(img_data))
                        temp_img = io.BytesIO()
                        image.save(temp_img, format="PNG")
                        temp_img.seek(0)
                        page_images.append((temp_img, current_page))
                        print(f"Successfully extracted image for EPUB page {current_page}")
                        logging.debug(f"Successfully extracted image for EPUB page {current_page}")
                        break
            return current_page, text, page_images

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, item in enumerate(html_items, 1):
                futures.append(executor.submit(extract_epub_page, item, i))
            for future in concurrent.futures.as_completed(futures):
                if stop_scanning:
                    app.update_progress(0, len(html_items), "Scanning stopped by user.", None, None)
                    break
                result = future.result()
                if result:
                    page_num, text, page_images = result
                    app.update_progress(page_num - 1, len(html_items), f"Extracting content for page {page_num} of {len(html_items)}...", None, None)
                    text_pages[page_num] = text
                    image_tasks.extend(page_images)

        total_pages = page_num
        if total_pages == 0:
            raise ValueError("No HTML content pages found in the EPUB file.")

        temp_pdf_path = os.path.join(output_dir, f"{epub_base_name}_temp.pdf")
        c = canvas.Canvas(temp_pdf_path, pagesize=letter)
        page_mappings = []
        current_pdf_page = 0
        saved_pages = set()

        for chunk_start in range(0, total_pages, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
            app.update_progress(chunk_start, total_pages, f"Processing pages {chunk_start + 1} to {chunk_end} of {total_pages}...", None, None)

            text_results = {}
            for page_num in range(chunk_start + 1, chunk_end + 1):
                if stop_scanning:
                    app.update_progress(page_num - 1, total_pages, "Scanning stopped by user.", None, None)
                    break
                text = text_pages.get(page_num, "")
                results, used_cache = grok_scan_content(text, is_image=False, challenge_report=app.challenge_report_var.get())
                text_results[page_num] = results
                cache_msg = " (cached)" if used_cache else ""
                print(f"Page {page_num} text flagged_content: {results}")
                logging.debug(f"Page {page_num} text flagged_content: {results}")
                app.update_progress(page_num - 1, total_pages, f"Scanning text for page {page_num}{cache_msg}...", None, results)

            image_results = defaultdict(list)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(grok_scan_content, task[0], is_image=True, challenge_report=app.challenge_report_var.get()): task for task in image_tasks if task[1] in range(chunk_start + 1, chunk_end + 1)}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    page_num = task[1]
                    try:
                        result, used_cache = future.result()
                        image_results[page_num].append(result)
                        cache_msg = " (cached)" if used_cache else ""
                        app.update_progress(page_num - 1, total_pages, f"Scanning image for page {page_num}{cache_msg}...", task[0], result)
                        print(f"Page {page_num} image flagged_content: {result}")
                        logging.debug(f"Page {page_num} image flagged_content: {result}")
                    except Exception as e:
                        print(f"Image scan failed for page {page_num}: {e}")
                        logging.error(f"Image scan failed for page {page_num}: {e}")
                        image_results[page_num].append([])

            for page_num in range(chunk_start + 1, chunk_end + 1):
                if stop_scanning:
                    app.update_progress(page_num, total_pages, "Scanning stopped by user.", None, None)
                    break
                text = text_pages.get(page_num, "")
                text_flags = text_results.get(page_num, [])
                image_flags = [item for sublist in image_results.get(page_num, []) for item in sublist]
                flagged_content = text_flags + image_flags
                page_flagged_content[page_num] = flagged_content
                app.update_progress(page_num - 1, total_pages, f"Processing page {page_num} of {total_pages}...", None, flagged_content)

                if flagged_content and page_num not in saved_pages:
                    start_pdf_page = current_pdf_page
                    c.setFont("Helvetica", 12)
                    y = 750
                    for line in text.split("\n"):
                        if y < 50:
                            c.showPage()
                            current_pdf_page += 1
                            c.setFont("Helvetica", 12)
                            y = 750
                        c.drawString(50, y, line[:100])
                        y -= 15
                    c.showPage()
                    current_pdf_page += 1
                    page_mappings.append((page_num, start_pdf_page, current_pdf_page - 1, text_flags))

        c.save()
        if not os.path.exists(temp_pdf_path):
            raise FileNotFoundError(f"Temporary PDF not created: {temp_pdf_path}")
        
        pdf_doc = fitz.open(temp_pdf_path)
        for epub_page_num, start_pdf_page, end_pdf_page, text_flags in page_mappings:
            text_flags = text_results.get(epub_page_num, [])
            image_flags = [item for sublist in image_results.get(epub_page_num, []) for item in sublist]
            flagged_content = text_flags + image_flags
            
            if flagged_content and epub_page_num not in saved_pages:
                for pdf_page_num in range(start_pdf_page, end_pdf_page + 1):
                    if pdf_page_num < len(pdf_doc):
                        page = pdf_doc[pdf_page_num]
                        for term in flatten_terms(text_flags):
                            for rect in page.search_for(term):
                                highlight = page.add_highlight_annot(rect)
                                highlight.set_colors((1, 0, 0, 0.5))
                                highlight.update()
                        pix = page.get_pixmap(dpi=200)
                        if pix.n > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        if pix.width == 0 or pix.height == 0 or not pix.samples:
                            print(f"Invalid pixmap for page {epub_page_num}")
                            logging.warning(f"Invalid pixmap for page {epub_page_num}")
                            continue
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.Resampling.LANCZOS)
                        img_path = os.path.join(output_dir, f"{epub_base_name}_page_{epub_page_num}.png")
                        img.save(img_path, optimize=True, quality=95)
                        output_images.append(img_path)
                        temp_image_paths.add(img_path)
                        saved_pages.add(epub_page_num)
                        all_flagged_content.extend(flagged_content)

        pdf_doc.save(temp_pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        pdf_doc.close()
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                print(f"Failed to remove temporary PDF {temp_pdf_path}: {str(e)}")
                logging.error(f"Failed to remove temporary PDF {temp_pdf_path}: {str(e)}")

        # Sort output_images by page number in Challenge mode
        if app.challenge_report_var.get():
            output_images_with_pages = []
            for img_path in output_images:
                page_num_str = img_path.split('_page_')[-1].replace('.png', '')
                page_num = int(page_num_str)
                output_images_with_pages.append((page_num, img_path))
            output_images_with_pages.sort(key=lambda x: x[0])
            output_images = [path for _, path in output_images_with_pages]
            sorted_page_flagged_content = {}
            for page_num, _ in output_images_with_pages:
                sorted_page_flagged_content[page_num] = page_flagged_content.get(page_num, [])
            page_flagged_content = sorted_page_flagged_content

        flagged_page_width, flagged_page_height = None, None
        if output_images:
            first_flagged_img = Image.open(output_images[0])
            flagged_page_width, flagged_page_height = first_flagged_img.size
            print(f"Flagged page dimensions (from saved image): {flagged_page_width}x{flagged_page_height}")
            logging.debug(f"Flagged page dimensions (from saved image): {flagged_page_width}x{flagged_page_height}")
        else:
            flagged_page_width, flagged_page_height = 612, 792
            print("No flagged images generated, using fallback dimensions: 612x792")
            logging.debug("No flagged images generated, using fallback dimensions: 612x792")

        return None, cover_image, total_pages, output_images, all_flagged_content, None, epub_base_name, image_tasks, flagged_page_height, flagged_page_width, None, text_pages, page_flagged_content, temp_image_paths
    
    except Exception as e:
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as remove_error:
                print(f"Failed to remove temporary PDF after error {temp_pdf_path}: {str(remove_error)}")
                logging.error(f"Failed to remove temporary PDF after error {temp_pdf_path}: {str(remove_error)}")
        app.update_progress(0, 0, f"Error: {str(e)}", None, None)
        logging.error(f"Error in process_epub: {e}")
        return None, None, 0, None, None, None, None, None, None, None, f"Error processing EPUB: {str(e)}", {}, {}, set()
		
def pack_images_to_pdf(image_paths, output_dir, base_name, app=None, include_report=False, report_img=None, temp_image_paths=None):
    if not image_paths:
        print("No images to pack into PDF.")
        return None, 0
    try:
        pdf_path = os.path.join(output_dir, f"{base_name}_flagged_content.pdf")
        pdf_doc = fitz.open()

        # Add all images in image_paths (including index pages, report, and flagged pages)
        for i, img_path in enumerate(image_paths):
            # Skip if this is the report path (already included in image_paths)
            if "_report.png" in img_path:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                temp_img = io.BytesIO()
                img.save(temp_img, format="PNG", optimize=True, quality=95)
                temp_img.seek(0)
                pdf_page = pdf_doc.new_page(width=img.width, height=img.height)
                pdf_page.insert_image(pdf_page.rect, stream=temp_img.read(), keep_proportion=True)
                temp_img.close()
            else:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                resized_img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.Resampling.LANCZOS)

                draw = ImageDraw.Draw(resized_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except Exception:
                    font = ImageFont.load_default().font_variant(size=20)
                page_num = int(os.path.basename(img_path).split("_page_")[1].split(".")[0])
                text = f"Page {page_num}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                draw.text((resized_img.width - text_width - 10, resized_img.height - text_height - 10), text, font=font, fill="black")

                temp_img = io.BytesIO()
                resized_img.save(temp_img, format="PNG", optimize=True, quality=95)
                temp_img.seek(0)

                pdf_page = pdf_doc.new_page(width=resized_img.width, height=resized_img.height)
                pdf_page.insert_image(pdf_page.rect, stream=temp_img.read(), keep_proportion=True)
                temp_img.close()

            if app and app.pack_to_pdf_var.get():
                app.ticker_label.config(text=f"Packing PDF: Added page {i + 1} of {len(image_paths)}")
                app.pack_progress_bar["value"] = ((i + 1) / len(image_paths)) * 100  # Adjusted denominator
                app.root.update_idletasks()

        pdf_doc.save(pdf_path, garbage=4, deflate=True, clean=True)
        pdf_doc.close()
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"Saved packed PDF: {pdf_path} (Size: {pdf_size_mb:.2f} MB)")
        if app and app.pack_to_pdf_var.get():
            app.ticker_label.config(text=f"PDF Packing Complete: {pdf_path}")
            app.pack_progress_bar["value"] = 100

        print(f"Directory contents before deletion: {os.listdir(output_dir)}")
        all_paths_to_delete = set(image_paths) | (temp_image_paths if temp_image_paths else set())
        for img_path in all_paths_to_delete:
            try:
                os.remove(img_path)
                print(f"Deleted image file: {img_path}")
            except Exception as e:
                print(f"Failed to delete image file {img_path}: {str(e)}")
        print(f"Directory contents after deletion: {os.listdir(output_dir)}")

        return pdf_path, pdf_size_mb
    except Exception as e:
        print(f"Error packing PDF: {str(e)}")
        logging.error(f"Error packing PDF: {e}")
        return None, 0

def process_file(file_path, title, author, cover_image_path, output_dir, app, pack_to_pdf=False, include_report=False, challenge_report=False):
    logging.debug(f"Starting process_file: file_path={file_path}, title={title}, pack_to_pdf={pack_to_pdf}, include_report={include_report}, challenge_report={challenge_report}")
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        result = process_pdf(file_path, title, author, cover_image_path, output_dir, app)
    elif file_extension == ".epub":
        result = process_epub(file_path, title, author, cover_image_path, output_dir, app)
    else:
        app.update_progress(0, 0, "Unsupported file format.", None, None)
        return None, "Unsupported file format. Please select a PDF or EPUB file.", None

    # Unpack results from processing
    pdf_doc, cover_image, total_pages, output_images, all_flagged_content, _, base_name, image_tasks, flagged_page_width, flagged_page_height, error, text_pages, page_flagged_content, temp_image_paths = result
    logging.debug(f"process_file result: {result}")

    if error:
        logging.error(f"Error in processing file: {error}")
        return None, error, None

    # Ensure variables are initialized properly
    all_flagged_content = all_flagged_content if all_flagged_content is not None else []
    output_images = output_images if output_images is not None else []
    temp_image_paths = temp_image_paths if temp_image_paths is not None else set()

    # Update UI with flagged terms (from full scan)
    if all_flagged_content:
        app.all_flagged_terms.update(flatten_terms(all_flagged_content))
        app.ticker_label.config(text=f"Flagged Terms: {', '.join(sorted(app.all_flagged_terms)) if app.all_flagged_terms else 'None'}")

    # Sanitize title for file naming
    sanitized_title = title.replace(" ", "_").replace(":", "").replace("\\", "").replace("/", "").replace("?", "").replace("*", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "") if title else "Untitled"
    report_img = None

    # Predefined dictionary for index page tally
    OFFENSIVE_WORDS = {
        "abortion": {"category": "Sensitive Social Topics", "severity": 3},
        "abused": {"category": "Violence", "severity": 3},
        "acid trips": {"category": "Drug/Alcohol Use", "severity": 3},
        "aggravated assault": {"category": "Violence", "severity": 4},
        "alcohol": {"category": "Drug/Alcohol Use", "severity": 2},
        "alcohol abuse": {"category": "Drug/Alcohol Use", "severity": 3},
        "alcoholic": {"category": "Drug/Alcohol Use", "severity": 3},
        "alcoholism": {"category": "Drug/Alcohol Use", "severity": 3},
        "america is a failed state": {"category": "Political/Ideological", "severity": 3},
        "ammunition": {"category": "Violence", "severity": 2},
        "anal sex": {"category": "Sexual Content", "severity": 4},
        "anger rapists": {"category": "Violence", "severity": 5},
        "anti-american": {"category": "Political/Ideological", "severity": 3},
        "anti-religious sentiments": {"category": "Political/Ideological", "severity": 3},
        "antiestablishment": {"category": "Political/Ideological", "severity": 3},
        "anus": {"category": "Sexual Content", "severity": 3},
        "armed": {"category": "Violence", "severity": 3},
        "aroused": {"category": "Sexual Content", "severity": 3},
        "arousing": {"category": "Sexual Content", "severity": 3},
        "ass": {"category": "Vulgar Language", "severity": 2},
        "asshole": {"category": "Vulgar Language", "severity": 2},
        "assault": {"category": "Violence", "severity": 3},
        "bareback sex": {"category": "Sexual Content", "severity": 4},
        "bastard": {"category": "Vulgar Language", "severity": 2},
        "beer": {"category": "Drug/Alcohol Use", "severity": 2},
        "bestiality": {"category": "Abhorrent Words", "severity": 5},
        "bigots": {"category": "Derogatory/Racial Slang", "severity": 3},
        "binary gender": {"category": "Sensitive Social Topics", "severity": 2},
        "biological racial hierarchy": {"category": "Political/Ideological", "severity": 3},
        "birth gender": {"category": "Sensitive Social Topics", "severity": 2},
        "bi": {"category": "Sensitive Social Topics", "severity": 2},
        "bitch": {"category": "Derogatory/Racial Slang", "severity": 3},
        "black lives matter": {"category": "Political/Ideological", "severity": 3},
        "black tar": {"category": "Drug/Alcohol Use", "severity": 3},
        "blow job": {"category": "Sexual Content", "severity": 4},
        "blow you off": {"category": "Sexual Content", "severity": 3},
        "blowies": {"category": "Sexual Content", "severity": 4},
        "boobs": {"category": "Sexual Content", "severity": 3},
        "bourgeoisie": {"category": "Political/Ideological", "severity": 3},
        "brothels": {"category": "Sexual Content", "severity": 4},
        "brutalized": {"category": "Violence", "severity": 4},
        "bum fun": {"category": "Sexual Content", "severity": 4},
        "bumming": {"category": "Sexual Content", "severity": 4},
        "butch dykes": {"category": "Derogatory/Racial Slang", "severity": 4},
        "capitalism": {"category": "Political/Ideological", "severity": 2},
        "chemsex": {"category": "Drug/Alcohol Use", "severity": 4},
        "child molester": {"category": "Abhorrent Words", "severity": 5},
        "child molesters": {"category": "Abhorrent Words", "severity": 5},
        "child pornography": {"category": "Abhorrent Words", "severity": 5},
        "child predators": {"category": "Abhorrent Words", "severity": 5},
        "chink": {"category": "Derogatory/Racial Slang", "severity": 5},
        "cis": {"category": "Sensitive Social Topics", "severity": 2},
        "cis gay men": {"category": "Sensitive Social Topics", "severity": 2},
        "cis lesbians": {"category": "Sensitive Social Topics", "severity": 2},
        "cis women": {"category": "Sensitive Social Topics", "severity": 2},
        "cisgender": {"category": "Sensitive Social Topics", "severity": 2},
        "class struggle": {"category": "Political/Ideological", "severity": 3},
        "clit": {"category": "Sexual Content", "severity": 3},
        "clitoral orgasm": {"category": "Sexual Content", "severity": 3},
        "clitoral stimulation": {"category": "Sexual Content", "severity": 3},
        "clitoris": {"category": "Sexual Content", "severity": 3},
        "cocaine": {"category": "Drug/Alcohol Use", "severity": 3},
        "cock": {"category": "Vulgar Language", "severity": 2},
        "cock-sucking": {"category": "Sexual Content", "severity": 4},
        "cocksucker": {"category": "Vulgar Language", "severity": 3},
        "condom": {"category": "Sexual Content", "severity": 2},
        "condoms": {"category": "Sexual Content", "severity": 2},
        "communism": {"category": "Political/Ideological", "severity": 3},
        "comrade": {"category": "Political/Ideological", "severity": 2},
        "coon": {"category": "Derogatory/Racial Slang", "severity": 5},
        "crackhead": {"category": "Derogatory/Racial Slang", "severity": 3},
        "cracker": {"category": "Derogatory/Racial Slang", "severity": 3},
        "critical race": {"category": "Political/Ideological", "severity": 3},
        "critical race theorist": {"category": "Political/Ideological", "severity": 3},
        "critical race theory": {"category": "Political/Ideological", "severity": 3},
        "critical gender theory": {"category": "Sensitive Social Topics", "severity": 3},
        "cross-sex hormones": {"category": "Sensitive Social Topics", "severity": 3},
        "cum": {"category": "Sexual Content", "severity": 3},
        "cunt": {"category": "Vulgar Language", "severity": 3},
        "damn": {"category": "Vulgar Language", "severity": 2},
        "defaced flag": {"category": "Political/Ideological", "severity": 3},
        "demerol": {"category": "Drug/Alcohol Use", "severity": 3},
        "dental dams": {"category": "Sexual Content", "severity": 3},
        "dick": {"category": "Vulgar Language", "severity": 2},
        "dick-shaped": {"category": "Sexual Content", "severity": 3},
        "dickhead": {"category": "Vulgar Language", "severity": 2},
        "die": {"category": "Violence", "severity": 3},
        "dildos": {"category": "Sexual Content", "severity": 4},
        "douche": {"category": "Vulgar Language", "severity": 2},
        "drag queen": {"category": "Sensitive Social Topics", "severity": 2},
        "drag shows": {"category": "Sensitive Social Topics", "severity": 2},
        "drank": {"category": "Drug/Alcohol Use", "severity": 2},
        "drinking alcohol": {"category": "Drug/Alcohol Use", "severity": 2},
        "drinking heavily": {"category": "Drug/Alcohol Use", "severity": 3},
        "drugs": {"category": "Drug/Alcohol Use", "severity": 3},
        "dyke": {"category": "Derogatory/Racial Slang", "severity": 4},
        "ecstasy": {"category": "Drug/Alcohol Use", "severity": 3},
        "ejaculated": {"category": "Sexual Content", "severity": 3},
        "erect": {"category": "Sexual Content", "severity": 3},
        "erection": {"category": "Sexual Content", "severity": 3},
        "erotic": {"category": "Sexual Content", "severity": 3},
        "erotica": {"category": "Sexual Content", "severity": 3},
        "fag": {"category": "Derogatory/Racial Slang", "severity": 4},
        "faggy": {"category": "Derogatory/Racial Slang", "severity": 4},
        "faggot": {"category": "Derogatory/Racial Slang", "severity": 4},
        "fags": {"category": "Derogatory/Racial Slang", "severity": 4},
        "fascism": {"category": "Political/Ideological", "severity": 3},
        "feminism": {"category": "Political/Ideological", "severity": 2},
        "fist in my vagina": {"category": "Sexual Content", "severity": 5},
        "forcible intercourse": {"category": "Sexual Content", "severity": 5},
        "fuck": {"category": "Vulgar Language", "severity": 3},
        "fuck me": {"category": "Sexual Content", "severity": 4},
        "fuck me, bitch": {"category": "Sexual Content", "severity": 4},
        "fuck you": {"category": "Vulgar Language", "severity": 3},
        "fucked": {"category": "Vulgar Language", "severity": 3},
        "fucked me hard": {"category": "Sexual Content", "severity": 4},
        "fucked up": {"category": "Vulgar Language", "severity": 3},
        "fucking": {"category": "Vulgar Language", "severity": 3},
        "gay": {"category": "Sensitive Social Topics", "severity": 2},
        "gaydar": {"category": "Sensitive Social Topics", "severity": 2},
        "gay porn": {"category": "Sexual Content", "severity": 4},
        "gay sex": {"category": "Sexual Content", "severity": 4},
        "gender theory": {"category": "Sensitive Social Topics", "severity": 3},
        "genital mutilation": {"category": "Sensitive Social Topics", "severity": 4},
        "genital surgery": {"category": "Sensitive Social Topics", "severity": 3},
        "get drunk or high": {"category": "Drug/Alcohol Use", "severity": 3},
        "glitter family": {"category": "Sensitive Social Topics", "severity": 2},
        "gook": {"category": "Derogatory/Racial Slang", "severity": 5},
        "gore": {"category": "Violence", "severity": 3},
        "gun": {"category": "Violence", "severity": 2},
        "gun control": {"category": "Political/Ideological", "severity": 2},
        "guns": {"category": "Violence", "severity": 2},
        "hand job": {"category": "Sexual Content", "severity": 4},
        "he/him": {"category": "Sensitive Social Topics", "severity": 2},
        "heroin": {"category": "Drug/Alcohol Use", "severity": 4},
        "high as kites": {"category": "Drug/Alcohol Use", "severity": 3},
        "homo": {"category": "Derogatory/Racial Slang", "severity": 4},
        "homophobes": {"category": "Sensitive Social Topics", "severity": 3},
        "human trafficking": {"category": "Abhorrent Words", "severity": 5},
        "humping": {"category": "Sexual Content", "severity": 4},
        "hypersexualized": {"category": "Sexual Content", "severity": 4},
        "inbreed": {"category": "Derogatory/Racial Slang", "severity": 3},
        "incest": {"category": "Abhorrent Words", "severity": 5},
        "intercourse": {"category": "Sexual Content", "severity": 3},
        "intersex": {"category": "Sensitive Social Topics", "severity": 2},
        "jerking off": {"category": "Sexual Content", "severity": 4},
        "jerkoff": {"category": "Vulgar Language", "severity": 2},
        "jizz": {"category": "Sexual Content", "severity": 3},
        "kegger party": {"category": "Drug/Alcohol Use", "severity": 3},
        "kike": {"category": "Derogatory/Racial Slang", "severity": 5},
        "kill": {"category": "Violence", "severity": 3},
        "kill him": {"category": "Violence", "severity": 4},
        "kill you": {"category": "Violence", "severity": 4},
        "kissed": {"category": "Sexual Content", "severity": 2},
        "knives": {"category": "Violence", "severity": 2},
        "knocked up": {"category": "Sexual Content", "severity": 3},
        "klub fuk": {"category": "Sexual Content", "severity": 4},
        "lesbian sex": {"category": "Sexual Content", "severity": 4},
        "lesbians": {"category": "Sensitive Social Topics", "severity": 2},
        "lgbtq": {"category": "Sensitive Social Topics", "severity": 2},
        "lgbthealth": {"category": "Sensitive Social Topics", "severity": 2},
        "lubricant": {"category": "Sexual Content", "severity": 3},
        "marginalized communities": {"category": "Political/Ideological", "severity": 2},
        "marijuana": {"category": "Drug/Alcohol Use", "severity": 3},
        "marxism": {"category": "Political/Ideological", "severity": 3},
        "masturbated": {"category": "Sexual Content", "severity": 4},
        "masturbating": {"category": "Sexual Content", "severity": 4},
        "masturbation": {"category": "Sexual Content", "severity": 4},
        "medically transition": {"category": "Sensitive Social Topics", "severity": 3},
        "meth": {"category": "Drug/Alcohol Use", "severity": 3},
        "molest": {"category": "Abhorrent Words", "severity": 5},
        "molested": {"category": "Abhorrent Words", "severity": 5},
        "molestation": {"category": "Abhorrent Words", "severity": 5},
        "molesters": {"category": "Abhorrent Words", "severity": 5},
        "multigender": {"category": "Sensitive Social Topics", "severity": 2},
        "murder": {"category": "Violence", "severity": 4},
        "mushrooms": {"category": "Drug/Alcohol Use", "severity": 3},
        "needle": {"category": "Drug/Alcohol Use", "severity": 3},
        "necrophilia": {"category": "Abhorrent Words", "severity": 5},
        "nigga": {"category": "Derogatory/Racial Slang", "severity": 5},
        "nigger": {"category": "Derogatory/Racial Slang", "severity": 5},
        "nipples": {"category": "Sexual Content", "severity": 3},
        "non-binary": {"category": "Sensitive Social Topics", "severity": 2},
        "nonstraight": {"category": "Sensitive Social Topics", "severity": 2},
        "nutsack": {"category": "Sexual Content", "severity": 3},
        "online meet-up sites for sex": {"category": "Sexual Content", "severity": 4},
        "oppressed": {"category": "Political/Ideological", "severity": 2},
        "oppressors": {"category": "Political/Ideological", "severity": 2},
        "oral sex": {"category": "Sexual Content", "severity": 4},
        "orgasm": {"category": "Sexual Content", "severity": 3},
        "orgasms": {"category": "Sexual Content", "severity": 3},
        "painkillers": {"category": "Drug/Alcohol Use", "severity": 3},
        "patriarchal society": {"category": "Political/Ideological", "severity": 2},
        "patriarchy": {"category": "Political/Ideological", "severity": 2},
        "pedophile": {"category": "Abhorrent Words", "severity": 5},
        "peed on me": {"category": "Sexual Content", "severity": 4},
        "penetrating": {"category": "Sexual Content", "severity": 4},
        "penetration": {"category": "Sexual Content", "severity": 4},
        "penis": {"category": "Sexual Content", "severity": 3},
        "phalloplasty": {"category": "Sensitive Social Topics", "severity": 3},
        "pills": {"category": "Drug/Alcohol Use", "severity": 3},
        "piss": {"category": "Vulgar Language", "severity": 2},
        "porn": {"category": "Sexual Content", "severity": 4},
        "predator": {"category": "Abhorrent Words", "severity": 5},
        "preferred gender": {"category": "Sensitive Social Topics", "severity": 2},
        "prick": {"category": "Vulgar Language", "severity": 2},
        "privatized education": {"category": "Political/Ideological", "severity": 2},
        "proletariat": {"category": "Political/Ideological", "severity": 3},
        "promiscuity": {"category": "Sexual Content", "severity": 3},
        "pronouns": {"category": "Sensitive Social Topics", "severity": 2},
        "protest fliers": {"category": "Political/Ideological", "severity": 2},
        "prostitutes": {"category": "Sexual Content", "severity": 4},
        "pussy": {"category": "Vulgar Language", "severity": 3},
        "pussy-eating": {"category": "Sexual Content", "severity": 4},
        "queer": {"category": "Sensitive Social Topics", "severity": 2},
        "racial equity": {"category": "Political/Ideological", "severity": 2},
        "racial essentialism": {"category": "Political/Ideological", "severity": 3},
        "racial justice": {"category": "Political/Ideological", "severity": 2},
        "racism": {"category": "Derogatory/Racial Slang", "severity": 4},
        "racist": {"category": "Derogatory/Racial Slang", "severity": 4},
        "radical extremist": {"category": "Political/Ideological", "severity": 3},
        "rainbow family": {"category": "Sensitive Social Topics", "severity": 2},
        "rape": {"category": "Abhorrent Words", "severity": 5},
        "raped": {"category": "Abhorrent Words", "severity": 5},
        "rapist's penis": {"category": "Abhorrent Words", "severity": 5},
        "rapists": {"category": "Abhorrent Words", "severity": 5},
        "religion class": {"category": "Political/Ideological", "severity": 2},
        "religious beliefs": {"category": "Political/Ideological", "severity": 2},
        "religious regime": {"category": "Political/Ideological", "severity": 3},
        "revenge": {"category": "Violence", "severity": 3},
        "same sex": {"category": "Sensitive Social Topics", "severity": 2},
        "same-sex": {"category": "Sensitive Social Topics", "severity": 2},
        "sex": {"category": "Sexual Content", "severity": 3},
        "sex change": {"category": "Sensitive Social Topics", "severity": 3},
        "sex changes": {"category": "Sensitive Social Topics", "severity": 3},
        "sex toy": {"category": "Sexual Content", "severity": 4},
        "sexual": {"category": "Sexual Content", "severity": 3},
        "sexual identity": {"category": "Sensitive Social Topics", "severity": 2},
        "sexual intercourse": {"category": "Sexual Content", "severity": 3},
        "sexualizing": {"category": "Sexual Content", "severity": 3},
        "sexuality": {"category": "Sensitive Social Topics", "severity": 2},
        "sexy": {"category": "Sexual Content", "severity": 3},
        "sexyfuntime": {"category": "Sexual Content", "severity": 4},
        "she/her": {"category": "Sensitive Social Topics", "severity": 2},
        "shit": {"category": "Vulgar Language", "severity": 2},
        "shoot": {"category": "Violence", "severity": 3},
        "shooting up": {"category": "Drug/Alcohol Use", "severity": 4},
        "skank": {"category": "Derogatory/Racial Slang", "severity": 3},
        "slaughter": {"category": "Violence", "severity": 4},
        "slimy asshole": {"category": "Vulgar Language", "severity": 3},
        "slut": {"category": "Derogatory/Racial Slang", "severity": 3},
        "sluts": {"category": "Derogatory/Racial Slang", "severity": 3},
        "smoked black tar": {"category": "Drug/Alcohol Use", "severity": 4},
        "social class distinctions": {"category": "Political/Ideological", "severity": 2},
        "social construct": {"category": "Political/Ideological", "severity": 2},
        "sodomy": {"category": "Sexual Content", "severity": 4},
        "spic": {"category": "Derogatory/Racial Slang", "severity": 5},
        "stab": {"category": "Violence", "severity": 3},
        "stabbed": {"category": "Violence", "severity": 3},
        "strap-ons": {"category": "Sexual Content", "severity": 4},
        "strip-club": {"category": "Sexual Content", "severity": 4},
        "stripper": {"category": "Sexual Content", "severity": 4},
        "structural racism": {"category": "Political/Ideological", "severity": 3},
        "suck dick": {"category": "Sexual Content", "severity": 4},
        "suck it": {"category": "Sexual Content", "severity": 4},
        "sucked him off": {"category": "Sexual Content", "severity": 4},
        "suicidal tendencies": {"category": "Sensitive Social Topics", "severity": 3},
        "suicide": {"category": "Sensitive Social Topics", "severity": 3},
        "systemic racism": {"category": "Political/Ideological", "severity": 3},
        "they/them": {"category": "Sensitive Social Topics", "severity": 2},
        "they/them pronouns": {"category": "Sensitive Social Topics", "severity": 2},
        "third gender": {"category": "Sensitive Social Topics", "severity": 2},
        "tits": {"category": "Sexual Content", "severity": 3},
        "titties": {"category": "Sexual Content", "severity": 3},
        "traditionally female roles": {"category": "Sensitive Social Topics", "severity": 2},
        "traffickers": {"category": "Abhorrent Words", "severity": 5},
        "tranny": {"category": "Derogatory/Racial Slang", "severity": 4},
        "trans sex": {"category": "Sexual Content", "severity": 4},
        "transgender": {"category": "Sensitive Social Topics", "severity": 3},
        "transgender ideology": {"category": "Sensitive Social Topics", "severity": 3},
        "transgender spectrum": {"category": "Sensitive Social Topics", "severity": 3},
        "transphobe": {"category": "Sensitive Social Topics", "severity": 3},
        "transphobes": {"category": "Sensitive Social Topics", "severity": 3},
        "transphobia": {"category": "Sensitive Social Topics", "severity": 3},
        "transphobic": {"category": "Sensitive Social Topics", "severity": 3},
        "transsexuals": {"category": "Sensitive Social Topics", "severity": 3},
        "transvestites": {"category": "Sensitive Social Topics", "severity": 2},
        "trisexual": {"category": "Sensitive Social Topics", "severity": 2},
        "tucking": {"category": "Sensitive Social Topics", "severity": 2},
        "twat": {"category": "Vulgar Language", "severity": 3},
        "twink": {"category": "Sensitive Social Topics", "severity": 2},
        "used needle": {"category": "Drug/Alcohol Use", "severity": 3},
        "vagina": {"category": "Sexual Content", "severity": 3},
        "vaginal": {"category": "Sexual Content", "severity": 3},
        "vaginal sex": {"category": "Sexual Content", "severity": 3},
        "vaginoplasty": {"category": "Sensitive Social Topics", "severity": 3},
        "valium": {"category": "Drug/Alcohol Use", "severity": 3},
        "vibrators": {"category": "Sexual Content", "severity": 4},
        "violence": {"category": "Violence", "severity": 2},
        "virgin": {"category": "Sexual Content", "severity": 2},
        "virginity": {"category": "Sexual Content", "severity": 2},
        "wank": {"category": "Sexual Content", "severity": 4},
        "weak little bitch": {"category": "Derogatory/Racial Slang", "severity": 3},
        "we sucked each other's dicks": {"category": "Sexual Content", "severity": 4},
        "wetback": {"category": "Derogatory/Racial Slang", "severity": 5},
        "white privilege": {"category": "Political/Ideological", "severity": 3},
        "whiteness": {"category": "Political/Ideological", "severity": 3},
        "whore": {"category": "Derogatory/Racial Slang", "severity": 3},
        "wine": {"category": "Drug/Alcohol Use", "severity": 2},
        "wrong puberty": {"category": "Sensitive Social Topics", "severity": 3},
        "ze/zir": {"category": "Sensitive Social Topics", "severity": 2}
    }

    # Store original flagged content for Challenge mode scoring
    original_page_flagged_content = {}
    if challenge_report:
        for page_num in text_pages:
            text = text_pages.get(page_num, "")
            text_results, _ = grok_scan_content(text, is_image=False, challenge_report=True)
            try:
                parsed_results = [FlaggedItem(**item) for item in text_results]
                original_page_flagged_content[page_num] = parsed_results
            except Exception as e:
                logging.error(f"Failed to parse Challenge mode text results for page {page_num}: {e}")
                original_page_flagged_content[page_num] = []
        for task in image_tasks:
            page_num = task[1]
            image_results, _ = grok_scan_content(task[0], is_image=True, challenge_report=True)
            try:
                parsed_results = [FlaggedItem(**item) for item in image_results]
                if page_num in original_page_flagged_content:
                    original_page_flagged_content[page_num].extend(parsed_results)
                else:
                    original_page_flagged_content[page_num] = parsed_results
            except Exception as e:
                logging.error(f"Failed to parse Challenge mode image results for page {page_num}: {e}")
                if page_num not in original_page_flagged_content:
                    original_page_flagged_content[page_num] = []

    # Decide pages to save
    pages_to_save = set()
    TARGET_CHALLENGE_PAGES = 10  # Target ~10 pages
    MAX_CHALLENGE_PAGES = 20    # Absolute maximum
    MIN_SCORE_THRESHOLD = 1.0   # Minimum score to include a page

    if challenge_report:
        # Scoring system for Challenge mode
        score_dict = {}
        for page_num, flagged_items in original_page_flagged_content.items():
            score_dict[page_num] = 0
            for item in flagged_items:
                severity = item.severity
                if severity == 5:
                    score_dict[page_num] += 5
                elif severity == 4:
                    score_dict[page_num] += 3
                elif severity == 3:
                    score_dict[page_num] += 1
                elif severity == 2:
                    score_dict[page_num] += 0.5
                else:
                    score_dict[page_num] += 0.1
                if item.category == "Egregious Content" or item.category == "Abhorrent Words":
                    score_dict[page_num] += 2

        # Sort and limit pages
        sorted_pages = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        eligible_pages = [(page_num, score) for page_num, score in sorted_pages if score >= MIN_SCORE_THRESHOLD]
        top_pages = eligible_pages[:TARGET_CHALLENGE_PAGES]
        if len(top_pages) < TARGET_CHALLENGE_PAGES and len(eligible_pages) > TARGET_CHALLENGE_PAGES:
            top_pages = eligible_pages[:min(MAX_CHALLENGE_PAGES, len(eligible_pages))]
        pages_to_save.update(page_num for page_num, _ in top_pages)
        logging.debug(f"Challenge mode: Selected {len(pages_to_save)} pages with scores: {dict(top_pages)}")
    else:
        # Standard mode: flag all pages with content from grok_scan_content
        for page_num, flagged_items in page_flagged_content.items():
            if flagged_items:
                pages_to_save.add(page_num)

    # Update output_images based on pages_to_save
    if challenge_report:
        output_images = [img for img in output_images if int(img.split('_page_')[-1].replace('.png', '')) in pages_to_save]

    # Compute tally for index page using OFFENSIVE_WORDS, counting terms within phrases
    category_tally = defaultdict(lambda: defaultdict(int))
    for page_num, text in text_pages.items():
        text_lower = text.lower()
        for word, info in OFFENSIVE_WORDS.items():
            # Count occurrences of the word, even within phrases
            count = sum(1 for _ in re.finditer(r'\b' + re.escape(word) + r'\b|\S*' + re.escape(word) + r'\S*', text_lower))
            if count > 0:
                category_tally[info["category"]][word] += count

    # Always generate index page, even with empty tally
    logging.debug(f"Generating index page for {len(output_images)} images with {len(category_tally)} categories")
    index_paths = generate_index_page(
        title, author, cover_image, category_tally, flagged_page_width, flagged_page_height,
        output_dir, sanitized_title, temp_image_paths, challenge_report=challenge_report
    )
    if index_paths:
        output_images[0:0] = index_paths
        logging.debug(f"Index pages added: {index_paths}")
    else:
        logging.warning("Failed to generate index page")
        index_paths = []  # Define index_paths with empty list if generation fails

    # Generate report if requested
    if include_report and all_flagged_content:
        app.update_progress(total_pages, total_pages, "Generating content analysis report...", None, None)
        report_img = generate_grok_report(all_flagged_content, title, author, flagged_page_width, flagged_page_height)
        if report_img:
            report_path = os.path.join(output_dir, f"{base_name}_report.png")
            report_img.save(report_path)
            output_images.insert(len(index_paths), report_path)
            temp_image_paths.add(report_path)
            print(f"Report generated and saved at {report_path}")

    # Pack to PDF if requested
    if output_images:
        app.update_progress(total_pages, total_pages, "Scanning completed. Packing PDF...", None, None)
        print(f"Preparing to pack {len(output_images)} images into PDF")
        if pack_to_pdf:
            pdf_path, pdf_size_mb = pack_images_to_pdf(output_images, output_dir, sanitized_title, app, include_report, report_img, temp_image_paths)
            app.pack_progress_bar.pack_forget()
            app.pack_progress_label.pack_forget()
            print(f"PDF packed at {pdf_path}")
            logging.debug(f"PDF packed: path={pdf_path}, size={pdf_size_mb}")
            return output_images, list(set(flatten_terms(all_flagged_content))), (pdf_path, pdf_size_mb)
        logging.debug("Images saved without packing to PDF")
        return output_images, list(set(flatten_terms(all_flagged_content))), None
    else:
        app.update_progress(total_pages, total_pages, "No flagged content found.", None, None)
        logging.debug("No flagged content found")
        return None, "No flagged content (text or images) found in any page.", None
		
class PDFScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BLOCKADE PDF/EPUB Content Scanner")
        self.root.geometry("800x850")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Scanning.Horizontal.TProgressbar", troughcolor="light grey", background="green")
        style.configure("Packing.Horizontal.TProgressbar", troughcolor="light grey", background="orange")

        self.file_path = None
        self.output_dir = output_dir
        self.scanning_thread = None
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.cover_image_path = None
        self.pack_to_pdf_var = tk.BooleanVar(value=True)
        self.include_report_var = tk.BooleanVar(value=False)
        self.challenge_report_var = tk.BooleanVar(value=False)
        self.turbo_mode_var = tk.BooleanVar(value=False)  # Turbo Mode variable
        self.current_image = None
        self.start_time = None
        self.timer_running = False
        self.all_flagged_terms = set()
        self.preview_update_id = None

        self.timer_label = tk.Label(root, text="Time Elapsed: 00:00", font=("Arial", 10))
        self.timer_label.pack(side=tk.TOP, anchor="nw", padx=10, pady=2)
        self.create_tooltip(self.timer_label, "Shows the time elapsed during scanning.")

        main_frame = tk.Frame(root)
        main_frame.pack(pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(left_frame, text="B.L.O.C.K.A.D.E.\nPDF/EPUB Content Scanner", font=("Arial", 16)).pack(pady=10)
        self.select_btn = tk.Button(left_frame, text="Select File", command=self.select_file)
        self.select_btn.pack(pady=5)
        self.create_tooltip(self.select_btn, "Choose a PDF or EPUB file to scan.")

        self.file_label = tk.Label(left_frame, text="No file selected", wraplength=300)
        self.file_label.pack(pady=5)
        self.create_tooltip(self.file_label, "Displays the currently selected file.")

        tk.Label(left_frame, text="Book Title:").pack(pady=5)
        self.title_entry = tk.Entry(left_frame, textvariable=self.title_var, width=30)
        self.title_entry.pack(pady=5)
        self.create_tooltip(self.title_entry, "Enter the title of the book (optional).")

        tk.Label(left_frame, text="Author:").pack(pady=5)
        self.author_entry = tk.Entry(left_frame, textvariable=self.author_var, width=30)
        self.author_entry.pack(pady=5)
        self.create_tooltip(self.author_entry, "Enter the author of the book (optional).")

        self.cover_btn = tk.Button(left_frame, text="Select Cover Image", command=self.select_cover_image)
        self.cover_btn.pack(pady=5)
        self.create_tooltip(self.cover_btn, "Select an optional cover image for the book.")

        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, padx=10)

        logo_path = "robot_input.png"
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).convert("RGBA")
            logo_img = logo_img.resize((150, 150), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(right_frame, image=self.logo_photo, bg="#f0f0f0")
            logo_label.pack(pady=5)
            self.create_tooltip(logo_label, "Application logo.")

        tk.Label(right_frame, text="Output Options:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
        self.save_images_radio = tk.Radiobutton(right_frame, text="Save as Images", variable=self.pack_to_pdf_var, value=False, bg="#f0f0f0")
        self.save_images_radio.pack(anchor="w", pady=2)
        self.create_tooltip(self.save_images_radio, "Save flagged pages as individual PNG images.")

        self.pack_pdf_radio = tk.Radiobutton(right_frame, text="Pack to PDF", variable=self.pack_to_pdf_var, value=True, bg="#f0f0f0")
        self.pack_pdf_radio.pack(anchor="w", pady=2)
        self.create_tooltip(self.pack_pdf_radio, "Pack flagged pages into a single PDF file.")

        self.include_report_check = tk.Checkbutton(right_frame, text="Include Report", variable=self.include_report_var, bg="#f0f0f0")
        self.include_report_check.pack(anchor="w", pady=2)
        self.create_tooltip(self.include_report_check, "Include a content analysis report in the output.")

        self.challenge_report_check = tk.Checkbutton(right_frame, text="Challenge Report", variable=self.challenge_report_var, bg="#f0f0f0")
        self.challenge_report_check.pack(anchor="w", pady=2)
        self.create_tooltip(self.challenge_report_check, "Use detailed challenge mode for content analysis.")

        self.turbo_mode_check = tk.Checkbutton(right_frame, text="Turbo Mode (8 threads)", variable=self.turbo_mode_var, bg="#f0f0f0")
        self.turbo_mode_check.pack(anchor="w", pady=2)
        self.create_tooltip(self.turbo_mode_check, "Increase to 8 threads for faster processing. Recommended for 8+ core CPUs.")

        self.preview_label = tk.Label(right_frame, text="Current Image Preview", font=("Arial", 10), bg="#f0f0f0")
        self.preview_label.pack(pady=5)
        self.create_tooltip(self.preview_label, "Preview of the currently processed image.")

        self.preview_frame = tk.Frame(right_frame, width=150, height=150, bg="#f0f0f0")
        self.preview_frame.pack(pady=5)
        self.preview_frame.pack_propagate(False)
        self.image_preview = tk.Label(self.preview_frame, bg="#f0f0f0")
        self.image_preview.pack(expand=True)
        self.create_tooltip(self.image_preview, "Displays a thumbnail of the current image being scanned.")

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        self.scan_btn = tk.Button(control_frame, text="Scan File", command=self.start_scan, state="disabled")
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.scan_btn, "Start scanning the selected file.")

        self.stop_btn = tk.Button(control_frame, text="Stop Scanning", command=self.stop_scan, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.stop_btn, "Stop the current scanning process.")

        self.progress_bar = ttk.Progressbar(root, length=600, mode="determinate", style="Scanning.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)
        self.create_tooltip(self.progress_bar, "Shows the progress of the scanning process.")

        self.ticker_label = tk.Label(root, text="Flagged Terms: None", wraplength=600, justify="left", height=4)
        self.ticker_label.pack(pady=5)
        self.create_tooltip(self.ticker_label, "Lists terms flagged during the scan.")

        self.status_label = tk.Label(root, text="", wraplength=600)
        self.status_label.pack(pady=5)
        self.create_tooltip(self.status_label, "Displays the current status of the application.")

        self.pack_progress_label = tk.Label(root, text="Packing Progress:", wraplength=600)
        self.pack_progress_label.pack(pady=2)
        self.create_tooltip(self.pack_progress_label, "Shows progress when packing images into a PDF.")
        self.pack_progress_bar = ttk.Progressbar(root, length=600, mode="determinate", style="Packing.Horizontal.TProgressbar")
        self.pack_progress_bar.pack(pady=2)
        self.create_tooltip(self.pack_progress_bar, "Progress bar for PDF packing.")
        self.pack_progress_bar.pack_forget()
        self.pack_progress_label.pack_forget()

        self.result_frame = tk.Frame(root, height=300)
        self.result_frame.pack(pady=5, fill=tk.X)
        self.result_frame.pack_propagate(False)

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.image_label, "Displays the final processed image.")

        self.result_label = scrolledtext.ScrolledText(self.result_frame, width=70, height=15, wrap=tk.WORD, bg="#f0f0f0")
        self.result_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.create_tooltip(self.result_label, "Shows detailed results of the scan.")

    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget."""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(self.tooltip, text=text, justify="left", background="#ffffe0", relief="solid", borderwidth=1, font=("Arial", "8"))
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("PDF/EPUB Files", "*.pdf *.epub")])
        if self.file_path:
            self.file_label.config(text=f"Selected: {os.path.basename(self.file_path)}")
            self.scan_btn.config(state="normal")
        else:
            self.file_label.config(text="No file selected")
            self.scan_btn.config(state="disabled")

    def select_cover_image(self):
        self.cover_image_path = filedialog.askopenfilename(title="Select Book Cover Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.cover_image_path:
            messagebox.showinfo("Success", "Cover image selected.")
        else:
            messagebox.showwarning("Warning", "No cover image selected. First page/cover will be used.")

    def start_scan(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first!")
            return
        
        global stop_scanning
        stop_scanning = False
        self.status_label.config(text="Loading document...")
        self.ticker_label.config(text="Flagged Terms: None")
        self.all_flagged_terms.clear()
        self.progress_bar["value"] = 0
        self.pack_progress_bar["value"] = 0
        self.pack_progress_bar.pack_forget()
        self.pack_progress_label.pack_forget()
        self.scan_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.image_label.config(image="")
        self.result_label.delete(1.0, tk.END)
        self.root.update_idletasks()

        self.start_time = time.time()
        self.timer_running = True
        self.update_timer()

        self.scanning_thread = threading.Thread(target=self.run_scan, args=(self.title_var.get(), self.author_var.get(), self.cover_image_path, self.pack_to_pdf_var.get(), self.include_report_var.get()))
        self.scanning_thread.start()

    def stop_scan(self):
        global stop_scanning
        stop_scanning = True
        self.stop_btn.config(state="disabled")
        self.timer_running = False
        if self.preview_update_id:
            self.root.after_cancel(self.preview_update_id)
            self.preview_update_id = None

    def update_timer(self):
        if self.timer_running:
            elapsed_time = int(time.time() - self.start_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            self.timer_label.config(text=f"Time Elapsed: {minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)

    def run_scan(self, title, author, cover_image_path, pack_to_pdf, include_report):
        logging.debug(f"Starting run_scan: title={title}, pack_to_pdf={pack_to_pdf}, include_report={include_report}")
        if pack_to_pdf:
            self.pack_progress_label.pack(pady=2, before=self.result_frame)
            self.pack_progress_bar.pack(pady=2, before=self.result_frame)
            self.root.update_idletasks()
        img_paths, result, pdf_info = process_file(self.file_path, title, author, cover_image_path, self.output_dir, self, pack_to_pdf, include_report, self.challenge_report_var.get())
        self.root.after(0, self.finish_scan, img_paths, result, pdf_info)

    def schedule_preview_update(self, img):
        try:
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)
            self.image_preview.config(image=photo)
            self.image_preview.image = photo
            self.root.update()
            print(f"Scheduled preview update for image of size {img.size}")
        except Exception as e:
            print(f"Error in schedule_preview_update: {e}")

    def update_progress(self, current, total, message, current_image=None, flagged_content=None):
        def _update_progress_ui(current, total, message, current_image, flagged_content):
            if total > 0:
                progress = (current / total) * 100
                self.progress_bar["value"] = progress
            self.status_label.config(text=message)
            if flagged_content:
                self.all_flagged_terms.update(flatten_terms(flagged_content))
                cache_status = " (cached)" if "(cached)" in message else ""
                self.ticker_label.config(text=f"Flagged Terms{cache_status}: {', '.join(sorted(self.all_flagged_terms)) if self.all_flagged_terms else 'None'}")
            if current_image:
                try:
                    current_image.seek(0)
                    img = Image.open(current_image)
                    print(f"Previewing image for page {current}/{total} - Image size: {img.size}")
                    self.schedule_preview_update(img)
                except Exception as e:
                    print(f"Error updating preview for page {current}/{total}: {e}")
                    self.image_preview.config(image="", text=f"Preview error: {str(e)}")
        self.root.after(0, lambda: _update_progress_ui(current, total, message, current_image, flagged_content))

    def finish_scan(self, img_paths, result, pdf_info):
        logging.debug(f"Entering finish_scan: img_paths={img_paths}, result={result}, pdf_info={pdf_info}")
        self.timer_running = False
        end_time = time.time()
        elapsed_time = int(end_time - self.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        self.scan_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Scan Complete")
        if self.preview_update_id:
            self.root.after_cancel(self.preview_update_id)
            self.preview_update_id = None

        if img_paths:
            logging.debug(f"img_paths exists: {len(img_paths)} images")
            if self.pack_to_pdf_var.get():
                logging.debug(f"pdf_info check: {pdf_info}")
                if pdf_info and isinstance(pdf_info, tuple) and len(pdf_info) == 2 and pdf_info[0] is not None:
                    pdf_path, pdf_size_mb = pdf_info
                    logging.debug(f"PDF info unpacked: path={pdf_path}, size={pdf_size_mb}")
                    result_text = f"Found: {', '.join(result)}\nPacked PDF: {pdf_path}"
                    self.result_label.delete(1.0, tk.END)
                    self.result_label.insert(tk.END, result_text)
                    messagebox.showinfo("Success", f"Images packed into PDF:\n{pdf_path}\nSize: {pdf_size_mb:.2f} MB\n\nYour scan finished in {minutes} minutes, {seconds} seconds.")
                else:
                    logging.warning(f"PDF packing failed or pdf_info invalid: {pdf_info}")
                    result_text = f"Found: {', '.join(result)}\nError: Failed to pack PDF."
                    self.result_label.delete(1.0, tk.END)
                    self.result_label.insert(tk.END, result_text)
                    messagebox.showwarning("Warning", f"PDF packing failed.\n\nYour scan finished in {minutes} minutes, {seconds} seconds.")
            else:
                try:
                    img = Image.open(img_paths[0])
                    img.thumbnail((500, 300))
                    photo = ImageTk.PhotoImage(img)
                    self.image_label.config(image=photo)
                    self.image_label.image = photo
                    result_text = f"Found: {', '.join(result)}\nSaved images:\n" + "\n".join(img_paths)
                    self.result_label.delete(1.0, tk.END)
                    self.result_label.insert(tk.END, result_text)
                    messagebox.showinfo("Success", f"Highlighted images saved:\n{', '.join(img_paths)}\n\nYour scan finished in {minutes} minutes, {seconds} seconds.")
                except Exception as e:
                    print(f"Error displaying final preview: {e}")
                    logging.error(f"Error displaying final preview: {e}")
                    self.image_label.config(image="")
                    result_text = f"Found: {', '.join(result)}\nSaved images:\n" + "\n".join(img_paths)
                    self.result_label.delete(1.0, tk.END)
                    self.result_label.insert(tk.END, result_text)
                    messagebox.showinfo("Success", f"Highlighted images saved:\n{', '.join(img_paths)}\n\nYour scan finished in {minutes} minutes, {seconds} seconds.")
        else:
            logging.debug("No img_paths returned")
            self.result_label.delete(1.0, tk.END)
            self.result_label.insert(tk.END, result)
			
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFScannerApp(root)
    root.mainloop()
