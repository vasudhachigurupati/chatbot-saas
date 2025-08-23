import os
import asyncio
import aiohttp
import sqlite3
import smtplib
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.responses import PlainTextResponse

import warnings
import logging
import json
import re
import xml.etree.ElementTree as ET
import contextlib

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Form, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

from fastapi import HTTPException, Depends
from fastapi import BackgroundTasks
from pydantic import BaseModel, EmailStr, Field, field_validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from decouple import config

# Configuration
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYDANTIC_V2"] = "1"

# App Configuration
SECRET_KEY = config("SECRET_KEY", default="e0b6bdb66a5e43b590a8a4b4f0e9c5a5d723e8c44dcf4a90a8e1c973c3f4e15d")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
GROQ_API_KEY = config("GROQ_API_KEY")
DATABASE_NAME = "chatbot_saas.db"

# Email Configuration
SMTP_SERVER = config("SMTP_SERVER", default="")
SMTP_PORT = int(config("SMTP_PORT", default="587"))
SMTP_USERNAME = config("SMTP_USERNAME", default="")
SMTP_PASSWORD = config("SMTP_PASSWORD", default="")
FROM_EMAIL = config("FROM_EMAIL", default=SMTP_USERNAME)
CONTACT_RECIPIENT = config("CONTACT_RECIPIENT", default=FROM_EMAIL)
FRONTEND_URL = config("FRONTEND_URL", default="http://localhost:5173")
# FIXED: Remove the os.getenv call from the .env file
BACKEND_API_BASE_URL = config("BACKEND_API_BASE_URL", default="http://localhost:8000")

# Subscription Plans
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free Plan",
        "max_chatbots": 1,
        "max_pages_per_bot": 10,
        "max_conversations_per_month": 100,
        "price": 0
    },
    "starter": {
        "name": "Starter Plan",
        "max_chatbots": 3,
        "max_pages_per_bot": 50,
        "max_conversations_per_month": 1000,
        "price": 29
    },
    "professional": {
        "name": "Professional Plan",
        "max_chatbots": 10,
        "max_pages_per_bot": 200,
        "max_conversations_per_month": 5000,
        "price": 99
    },
    "enterprise": {
        "name": "Enterprise Plan",
        "max_chatbots": -1,  # Unlimited
        "max_pages_per_bot": 1000,
        "max_conversations_per_month": -1,  # Unlimited
        "price": 299
    }
}

# Initialize services
groq_client = Groq(api_key=GROQ_API_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
def init_database():
    """Initialize the SQLite database with all required tables"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Users table with enhanced fields
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        company_name TEXT NOT NULL,
        plan TEXT NOT NULL DEFAULT 'free',
        is_active BOOLEAN DEFAULT FALSE,
        is_verified BOOLEAN DEFAULT FALSE,
        verification_token TEXT,
        verification_expires TIMESTAMP,
        reset_token TEXT,
        reset_expires TIMESTAMP,
        subscription_expires TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Enhanced chatbots table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chatbots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        website_url TEXT NOT NULL,
        sitemap_url TEXT,
        max_pages INTEGER DEFAULT 10,
        brand_voice TEXT DEFAULT 'friendly',
        system_prompt TEXT,
        is_active BOOLEAN DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'pending',
        embed_code TEXT,
        api_key TEXT UNIQUE,
        widget_settings TEXT DEFAULT '{}',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_trained TIMESTAMP,
        total_conversations INTEGER NOT NULL DEFAULT 0,
        monthly_conversations INTEGER NOT NULL DEFAULT 0,
        last_conversation_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    # Website content table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS website_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chatbot_id INTEGER NOT NULL,
        url TEXT NOT NULL,
        title TEXT,
        content TEXT,
        metadata TEXT,
        crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id) ON DELETE CASCADE
    )
    """)
    
    # Enhanced conversations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chatbot_id INTEGER NOT NULL,
        session_id TEXT,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        response_time_ms INTEGER,
        user_ip TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id) ON DELETE CASCADE
    )
    """)
    
    # Analytics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chatbot_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        event_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id) ON DELETE CASCADE
    )
    """)

    # Usage tracking table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usage_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        resource_type TEXT NOT NULL,
        usage_count INTEGER DEFAULT 1,
        period_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        period_end TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    # Contact messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contact_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        subject TEXT NOT NULL,
        message TEXT NOT NULL,
        user_id INTEGER,
        company_name TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contact_messages_created_at ON contact_messages(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contact_messages_status ON contact_messages(status)")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_db_connection():
    """Returns a new SQLite connection."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

@contextlib.contextmanager
def get_db():
    """Provides a database connection that automatically closes."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# Email utility functions
def send_email(to_email: str, subject: str, body: str, is_html: bool = False, reply_to: str | None = None):
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL]):
        logger.error("Email configuration is incomplete.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        if reply_to:
            msg.add_header('Reply-To', reply_to)
        msg.attach(MIMEText(body, 'html' if is_html else 'plain'))

        with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT)) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD.replace(" ", ""))  # strip spaces
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())

        logger.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logger.exception(f"Email send failed to {to_email}: {e}")
        return False

def generate_verification_token():
    """Generate a secure verification token"""
    return secrets.token_urlsafe(32)

def generate_api_key():
    """Generate a unique API key for chatbot"""
    return f"cb_{secrets.token_urlsafe(32)}"

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    company_name: str
    plan: str = Field(default="free")
    
    @field_validator('plan')
    @classmethod
    def validate_plan(cls, v):
        if v not in SUBSCRIPTION_PLANS.keys():
            raise ValueError(f'Plan must be one of: {", ".join(SUBSCRIPTION_PLANS.keys())}')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class EmailVerification(BaseModel):
    token: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=6)

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str
    user_id: Optional[int] = None # Added for logged-in users

class ResetPasswordForm(BaseModel):
    token: str
    new_password: str

class ChatbotCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    website_url: str
    sitemap_url: Optional[str] = None
    max_pages: int = Field(default=10, ge=1, le=1000)
    brand_voice: str = Field(default="friendly")
    
    @field_validator('website_url')
    @classmethod
    def validate_website_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Website URL must start with http:// or https://')
        return v
    
    @field_validator('brand_voice')
    @classmethod
    def validate_brand_voice(cls, v):
        if v not in ['friendly', 'professional', 'casual', 'technical']:
            raise ValueError('Brand voice must be one of: friendly, professional, casual, technical')
        return v

class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str

class WidgetSettings(BaseModel):
    primary_color: str = Field(default="#667eea")
    position: str = Field(default="bottom-right")
    greeting_message: str = Field(default="Hello! How can I help you?")
    placeholder_text: str = Field(default="Type your message...")

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate user from JWT token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="No credentials provided")
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Get user from database
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ? AND is_active = 1 AND is_verified = 1", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        raise HTTPException(status_code=401, detail="User not found or not verified")
    
    return {
        "id": user[0],
        "email": user[1],
        "company_name": user[3],
        "plan": user[4],
        "is_active": user[5],
        "is_verified": user[6],
        "subscription_expires": user[11]
    }

def check_usage_limits(user: Dict, resource_type: str, current_usage: int = 0) -> bool:
    """Check if user has exceeded usage limits"""
    plan_limits = SUBSCRIPTION_PLANS.get(user["plan"], SUBSCRIPTION_PLANS["free"])
    
    if resource_type == "chatbots":
        max_allowed = plan_limits["max_chatbots"]
        return max_allowed == -1 or current_usage < max_allowed
    elif resource_type == "pages":
        max_allowed = plan_limits["max_pages_per_bot"]
        return max_allowed == -1 or current_usage <= max_allowed
    elif resource_type == "conversations":
        max_allowed = plan_limits["max_conversations_per_month"]
        return max_allowed == -1 or current_usage < max_allowed
    
    return True

def update_conversation_count(chatbot_id: int, conn: sqlite3.Connection):
    """Update monthly conversation count and reset if needed"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT monthly_conversations, last_conversation_reset 
        FROM chatbots WHERE id = ?
    """, (chatbot_id,))
    
    result = cursor.fetchone()
    if result:
        monthly_count, last_reset = result
        last_reset_date = datetime.fromisoformat(last_reset)
        current_date = datetime.now()
        
        if (current_date.year != last_reset_date.year or 
            current_date.month != last_reset_date.month):
            monthly_count = 0
            cursor.execute("""
                UPDATE chatbots 
                SET monthly_conversations = 0, last_conversation_reset = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chatbot_id,))
            
        cursor.execute("""
            UPDATE chatbots 
            SET total_conversations = total_conversations + 1,
                monthly_conversations = monthly_conversations + 1
            WHERE id = ?
        """, (chatbot_id,))
        
        conn.commit()
        return monthly_count + 1
    return 0

class WebScraper:
    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages
        self.visited_urls = set()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; WebScraper/1.0)'}
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def can_fetch(self, url: str) -> bool:
        """Check robots.txt compliance"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        return text.strip()
    
    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract meaningful content from HTML"""
        try:
            if not html:
                return {}
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content with priority selectors
            content_selectors = [
                'main', 'article', '.content', '#content',
                '.main-content', '.post-content', '.entry-content'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text()
                    break
            
            if not main_content:
                body = soup.find('body')
                main_content = body.get_text() if body else soup.get_text()
            
            cleaned_content = self.clean_text(main_content)
            
            # Skip pages with insufficient content
            if len(cleaned_content) < 50:
                logger.warning(f"Insufficient content from {url}: {len(cleaned_content)} characters")
                return {}
            
            # Extract metadata
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ''
            
            # Extract headings
            headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if h.get_text().strip()]
            
            return {
                "url": url,
                "title": title_text,
                "content": cleaned_content,
                "description": description,
                "headings": headings,
                "word_count": len(cleaned_content.split())
            }
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {}

    async def get_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        """Extract URLs from XML sitemap"""
        urls = []
        try:
            logger.info(f"Fetching sitemap: {sitemap_url}")
            async with self.session.get(sitemap_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch sitemap: HTTP {response.status}")
                    return []
                    
                content = await response.text()
                root = ET.fromstring(content)
                
                # Handle namespaced and non-namespaced XML
                namespaces = {'': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                # Try with namespace first
                for url_elem in root.findall('.//url', namespaces):
                    loc_elem = url_elem.find('.//loc', namespaces)
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())
                
                # Fallback without namespace
                if not urls:
                    for url_elem in root.findall('.//url'):
                        loc_elem = url_elem.find('.//loc')
                        if loc_elem is not None and loc_elem.text:
                            urls.append(loc_elem.text.strip())
                
                logger.info(f"Found {len(urls)} URLs in sitemap")
                return urls[:self.max_pages]
                
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
            return []

    async def discover_urls_by_crawling(self, base_url: str) -> List[str]:
        """Discover URLs by crawling website links"""
        urls_to_visit = [base_url]
        discovered_urls = []
        base_domain = urlparse(base_url).netloc
        
        while urls_to_visit and len(discovered_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            try:
                if not self.can_fetch(current_url):
                    logger.info(f"Robots.txt disallows: {current_url}")
                    continue
                
                async with self.session.get(current_url) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {current_url}")
                        continue
                        
                    html = await response.text()
                    self.visited_urls.add(current_url)
                    discovered_urls.append(current_url)
                    
                    # Extract more URLs from this page
                    soup = BeautifulSoup(html, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(current_url, href)
                        
                        # Only include URLs from same domain
                        if (urlparse(full_url).netloc == base_domain and 
                            full_url not in self.visited_urls and 
                            full_url not in urls_to_visit):
                            urls_to_visit.append(full_url)
                            
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                
        logger.info(f"Discovered {len(discovered_urls)} URLs by crawling")
        return discovered_urls

    async def get_urls_to_scrape(self, website_url: str, sitemap_url: Optional[str] = None) -> List[str]:
        """Unified method to get URLs - try sitemap first, then crawling"""
        urls = []
        
        # Try sitemap first if provided
        if sitemap_url:
            logger.info("Attempting to use sitemap")
            urls = await self.get_urls_from_sitemap(sitemap_url)
        
        # Fall back to crawling if no sitemap or sitemap failed
        if not urls:
            logger.info("Falling back to URL discovery by crawling")
            urls = await self.discover_urls_by_crawling(website_url)
            
        # Ensure we have at least the base URL
        if not urls:
            logger.info("Using base URL as fallback")
            urls = [website_url]
            
        return urls[:self.max_pages]

    async def scrape_content_from_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape content from list of URLs"""
        scraped_data = []
        
        for url in urls:
            try:
                logger.info(f"Scraping: {url}")
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        content_data = self.extract_content(html, url)
                        
                        if content_data and content_data.get("word_count", 0) > 50:
                            scraped_data.append(content_data)
                            logger.info(f"Successfully scraped: {url} ({content_data['word_count']} words)")
                        else:
                            logger.warning(f"Insufficient content from: {url}")
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        logger.info(f"Successfully scraped {len(scraped_data)} pages")
        return scraped_data

def generate_system_prompt(brand_voice: str, website_content: List[Dict]) -> str:
    """Generate system prompt based on content and brand voice"""
    voice_instructions = {
        "friendly": "You are a friendly and approachable assistant. Use a warm, conversational tone and be helpful and enthusiastic.",
        "professional": "You are a professional assistant. Use formal language, be precise and informative, and maintain a business-appropriate tone.",
        "casual": "You are a casual and relaxed assistant. Use informal language, be conversational and easy-going.",
        "technical": "You are a technical expert. Use precise terminology, provide detailed explanations, and focus on accuracy and technical depth."
    }
    
    content_summary = []
    for content in website_content[:5]:
        content_summary.append(f"Page: {content['title']}\nContent: {content['content'][:500]}...")
    
    return f"""You are a helpful assistant for a website. {voice_instructions.get(brand_voice, voice_instructions['friendly'])}

Based on the following website content, answer user questions accurately and helpfully:

{chr(10).join(content_summary)}

Guidelines:
1. Only answer questions related to the website content provided
2. If you don't know the answer, say so politely
3. Be helpful and direct users to relevant sections
4. Maintain the {brand_voice} tone throughout
5. For topics not covered, explain you can only help with this website's content"""

async def generate_ai_response(message: str, system_prompt: str, conversation_history: List[Dict] = None) -> str:
    """Generate AI response using Groq"""
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            for conv in conversation_history[-5:]:
                messages.append({"role": "user", "content": conv["user_message"]})
                messages.append({"role": "assistant", "content": conv["bot_response"]})
        
        messages.append({"role": "user", "content": message})
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1000,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

async def process_website(chatbot_id: int, chatbot_data: ChatbotCreate):
    try:
        logger.info(f"Starting to process chatbot {chatbot_id}")
        
        async with WebScraper(max_pages=chatbot_data.max_pages) as scraper:
            urls_to_scrape = await scraper.get_urls_to_scrape(
                chatbot_data.website_url, 
                chatbot_data.sitemap_url
            )
            
            if not urls_to_scrape:
                raise Exception("No URLs could be discovered for scraping")
            
            scraped_content = await scraper.scrape_content_from_urls(urls_to_scrape)
            
            if not scraped_content:
                raise Exception("No content could be scraped from the website")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("DELETE FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
                
                for content in scraped_content:
                    cursor.execute("""
                        INSERT INTO website_content (chatbot_id, url, title, content, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        chatbot_id,
                        content["url"],
                        content["title"],
                        content["content"],
                        json.dumps({
                            "description": content["description"],
                            "headings": content["headings"],
                            "word_count": content["word_count"]
                        })
                    ))
                
                system_prompt = generate_system_prompt(chatbot_data.brand_voice, scraped_content)
                api_key = generate_api_key()
                embed_code = generate_embed_code(chatbot_id, api_key)
                
                cursor.execute("""
                UPDATE chatbots 
                SET system_prompt = ?, status = 'active', updated_at = CURRENT_TIMESTAMP, 
                    last_trained = CURRENT_TIMESTAMP, api_key = ?, embed_code = ?
                WHERE id = ?
                """, (system_prompt, api_key, embed_code, chatbot_id))
                
                conn.commit()
                logger.info(f"✅ Chatbot {chatbot_id} processed successfully with {len(scraped_content)} pages")
                
            except Exception as db_error:
                conn.rollback()
                logger.error(f"Database error for chatbot {chatbot_id}: {db_error}")
                raise
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error processing chatbot {chatbot_id}: {error_msg}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE chatbots 
                SET status = 'failed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chatbot_id,))
            conn.commit()

def generate_embed_code(chatbot_id: int, api_key: str) -> str:
    """Generate embed code for the chatbot widget"""
    return f"""
<div id="chatbot-widget-{chatbot_id}"></div>
<script>
    (function() {{
        var script = document.createElement('script');
        script.src = '{BACKEND_API_BASE_URL}/widget.js';
        script.async = true;
        script.onload = function() {{
            window.ChatbotWidget.init({{
                chatbotId: '{chatbot_id}',
                apiKey: '{api_key}',
                apiUrl: '{BACKEND_API_BASE_URL}'
            }});
        }};
        document.head.appendChild(script);
    }})();
</script>
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Enhanced Chatbot SaaS API is starting up...")
    init_database()
    logger.info("📊 Database initialized successfully!")
    yield
    logger.info("👋 Shutting down...")

app = FastAPI(
    title="Enhanced Chatbot SaaS API",
    description="A comprehensive SaaS platform for creating intelligent chatbots from any website",
    version="3.0.0",
    lifespan=lifespan
)

# FIXED: Better CORS configuration
frontend_url = config("FRONTEND_URL", default="http://localhost:5173").strip()
allowed_origins = config("ALLOWED_ORIGINS", default="").strip()

if allowed_origins:
    allowed = [o.strip() for o in allowed_origins.split(",") if o.strip()]
else:
    allowed = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    if frontend_url and "your-site" not in frontend_url:
        allowed.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

@app.get("/")
async def root():
    return {
        "message": "🤖 Enhanced Chatbot SaaS API",
        "version": "3.0.0",
        "docs": "/docs",
        "status": "active",
        "features": [
            "Email verification",
            "Subscription plans",
            "Usage tracking",
            "Embeddable widgets",
            "API access"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    email_configured = bool(SMTP_USERNAME and SMTP_PASSWORD)
    
    return {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "email_configured": email_configured,
        "version": "3.0.0"
    }

@app.post("/auth/register", response_model=dict)
async def register(user_data: UserCreate, background_tasks: BackgroundTasks):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            verification_token = generate_verification_token()
            verification_expires = datetime.now() + timedelta(hours=24)
            
            hashed_password = get_password_hash(user_data.password)
            cursor.execute("""
                INSERT INTO users (email, hashed_password, company_name, plan,
                                 verification_token, verification_expires)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_data.email, 
                  hashed_password,
                  user_data.company_name, 
                  user_data.plan,
                  verification_token, 
                  verification_expires.isoformat()))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            verification_link = f"{FRONTEND_URL}/?token={verification_token}"
            email_body = f"""
            <html>
            <body>
                <h2>Welcome to Chatbot SaaS!</h2>
                <p>Hi there,</p>
                <p>Thank you for registering with us. Please click the link below to verify your email address:</p>
                <p><a href="{verification_link}" style="background-color: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Verify Email</a></p>
                <p>This link will expire in 24 hours.</p>
                <p>If you didn't create this account, please ignore this email.</p>
                <p>Best regards,<br>The Chatbot SaaS Team</p>
            </body>
            </html>
            """
            
            background_tasks.add_task(send_email, user_data.email, "Verify Your Email - Chatbot SaaS", email_body, is_html=True)
            
            logger.info(f"New user registered: {user_data.email} (ID: {user_id})")
            
            return {
                "message": "Account created successfully. Please check your email to verify your account.",
                "user_id": user_id,
                "email": user_data.email
            }
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Error during user registration: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed due to an internal error.")

@app.post("/auth/verify-email")
async def verify_email(verification_data: EmailVerification):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, email, verification_expires FROM users 
                WHERE verification_token = ? AND is_verified = 0
            """, (verification_data.token,))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=400, detail="Invalid or expired verification token")
            
            # Check if token is expired
            expires_at = datetime.fromisoformat(user[2])
            if datetime.now() > expires_at:
                raise HTTPException(status_code=400, detail="Verification token has expired")
            
            # Verify user
            cursor.execute("""
                UPDATE users 
                SET is_verified = 1, is_active = 1, verification_token = NULL, 
                    verification_expires = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user[0],))
            
            conn.commit()
            
            logger.info(f"User verified: {user[1]} (ID: {user[0]})")
            
            return {"message": "Email verified successfully. You can now log in."}
        except Exception as e:
            conn.rollback()
            raise e

@app.post("/auth/resend-verification")
async def resend_verification(email_data: dict, background_tasks: BackgroundTasks):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            email = email_data.get("email")
            cursor.execute("""
                SELECT id FROM users 
                WHERE email = ? AND is_verified = 0
            """, (email,))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=400, detail="User not found or already verified")
            
            verification_token = generate_verification_token()
            verification_expires = datetime.now() + timedelta(hours=24)
            
            cursor.execute("""
                UPDATE users 
                SET verification_token = ?, verification_expires = ?
                WHERE id = ?
            """, (verification_token, verification_expires.isoformat(), user[0]))
            
            conn.commit()
            
            verification_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"
            email_body = f"""
            <html>
            <body>
                <h2>Email Verification - Chatbot SaaS</h2>
                <p>Hi there,</p>
                <p>You requested a new verification link. Please click the link below:</p>
                <p><a href="{verification_link}" style="background-color: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Verify Email</a></p>
                <p>This link will expire in 24 hours.</p>
            </body>
            </html>
            """
            
            background_tasks.add_task(send_email, email, "New Verification Link - Chatbot SaaS", email_body, is_html=True)
            
            return {"message": "Verification email sent successfully"}
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login", response_model=dict)
async def login(username: str = Form(...), password: str = Form(...)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, email, hashed_password, is_verified, is_active 
                FROM users WHERE email = ?
            """, (username,))
            user = cursor.fetchone()
            
            if not user or not verify_password(password, user[2]):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            if not user[3]:
                raise HTTPException(status_code=401, detail="Please verify your email address first")
            
            if not user[4]:
                raise HTTPException(status_code=401, detail="Account is deactivated")
            
            access_token = create_access_token(data={"sub": str(user[0]), "email": user[1]})
            
            logger.info(f"User logged in: {user[1]} (ID: {user[0]})")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user[0],
                    "email": user[1]
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/forgot-password")
async def forgot_password(reset_data: PasswordReset, background_tasks: BackgroundTasks):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM users WHERE email = ? AND is_verified = 1", (reset_data.email,))
            user = cursor.fetchone()
            
            if not user:
                return {"message": "If the email exists, a reset link has been sent."}
            
            reset_token = generate_verification_token()
            reset_expires = datetime.now() + timedelta(hours=2)
            
            cursor.execute("""
                UPDATE users 
                SET reset_token = ?, reset_expires = ?
                WHERE id = ?
            """, (reset_token, reset_expires.isoformat(), user[0]))
            
            conn.commit()
            
            reset_link = f"{FRONTEND_URL}/reset-password?reset_token={reset_token}"
            
            email_body = f"""
            <html>
            <body>
                <h2>Password Reset - Chatbot SaaS</h2>
                <p>Hi there,</p>
                <p>You requested a password reset. Click the link below to reset your password:</p>
                <p><a href="{reset_link}" style="background-color: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>This link will expire in 2 hours.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
            </html>
            """
            
            background_tasks.add_task(send_email, reset_data.email, "Password Reset - Chatbot SaaS", email_body, is_html=True)
            
            return {"message": "If the email exists, a reset link has been sent."}
            
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/reset-password")
async def reset_password(reset_data: PasswordResetConfirm):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, reset_expires FROM users 
                WHERE reset_token = ?
            """, (reset_data.token,))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=400, detail="Invalid reset token")
            
            expires_at = datetime.fromisoformat(user[1])
            if datetime.now() > expires_at:
                raise HTTPException(status_code=400, detail="Reset token has expired")
            
            hashed_password = get_password_hash(reset_data.new_password)
            cursor.execute("""
                UPDATE users 
                SET hashed_password = ?, reset_token = NULL, reset_expires = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (hashed_password, user[0]))
            
            conn.commit()
            
            return {"message": "Password reset successfully"}
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM chatbots WHERE user_id = ?", (current_user["id"],))
            chatbot_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT SUM(monthly_conversations) FROM chatbots WHERE user_id = ?
            """, (current_user["id"],))
            monthly_conversations = cursor.fetchone()[0] or 0
            
            plan_info = SUBSCRIPTION_PLANS.get(current_user["plan"], SUBSCRIPTION_PLANS["free"])
            
            return {
                **current_user,
                "stats": {
                    "chatbots": chatbot_count,
                    "monthly_conversations": monthly_conversations
                },
                "plan_limits": plan_info,
                "usage_percentage": {
                    "chatbots": (chatbot_count / plan_info["max_chatbots"] * 100) if plan_info["max_chatbots"] != -1 else 0,
                    "conversations": (monthly_conversations / plan_info["max_conversations_per_month"] * 100) if plan_info["max_conversations_per_month"] != -1 else 0
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbots/create")
async def create_chatbot(
    chatbot_data: ChatbotCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM chatbots WHERE user_id = ?", (current_user["id"],))
            current_count = cursor.fetchone()[0]
            
            if not check_usage_limits(current_user, "chatbots", current_count):
                plan_info = SUBSCRIPTION_PLANS.get(current_user["plan"])
                raise HTTPException(
                    status_code=403, 
                    detail=f"Chatbot limit reached. Your {plan_info['name']} allows {plan_info['max_chatbots']} chatbots."
                )
            
            if not check_usage_limits(current_user, "pages", chatbot_data.max_pages):
                plan_info = SUBSCRIPTION_PLANS.get(current_user["plan"])
                raise HTTPException(
                    status_code=403,
                    detail=f"Page limit exceeded. Your {plan_info['name']} allows {plan_info['max_pages_per_bot']} pages per chatbot."
                )
            
            cursor.execute("""
                INSERT INTO chatbots (user_id, name, website_url, sitemap_url, max_pages, brand_voice, status)
                VALUES (?, ?, ?, ?, ?, ?, 'processing')
            """, (
                current_user["id"],
                chatbot_data.name,
                chatbot_data.website_url,
                chatbot_data.sitemap_url,
                chatbot_data.max_pages,
                chatbot_data.brand_voice
            ))
            
            chatbot_id = cursor.lastrowid
            conn.commit()
            
            background_tasks.add_task(process_website, chatbot_id, chatbot_data)
            
            logger.info(f"Chatbot created: {chatbot_data.name} (ID: {chatbot_id}) for user {current_user['id']}")
            
            return {
                "id": chatbot_id,
                "name": chatbot_data.name,
                "status": "processing",
                "message": "Chatbot creation started. Website scraping in progress..."
            }
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Error during chatbot creation: {e}")
            raise HTTPException(status_code=500, detail="Chatbot creation failed due to an internal error.")

@app.get("/chatbots/list")
async def list_chatbots(current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, website_url, status, created_at, total_conversations,
                       monthly_conversations, api_key,
                       (SELECT COUNT(*) FROM website_content WHERE chatbot_id = chatbots.id) as page_count
                FROM chatbots 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            """, (current_user["id"],))
            
            chatbots = []
            for row in cursor.fetchall():
                chatbots.append({
                    "id": row[0],
                    "name": row[1],
                    "website_url": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "total_conversations": row[5],
                    "monthly_conversations": row[6],
                    "api_key": row[7],
                    "page_count": row[8]
                })
            
            return chatbots
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM chatbots 
                WHERE id = ? AND user_id = ?
            """, (chatbot_id, current_user["id"]))
            
            chatbot = cursor.fetchone()
            if not chatbot:
                raise HTTPException(status_code=404, detail="Chatbot not found")
            
            cursor.execute("SELECT COUNT(*) FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
            content_count = cursor.fetchone()[0]
            
            return {
                "id": chatbot[0],
                "name": chatbot[2],
                "website_url": chatbot[3],
                "sitemap_url": chatbot[4],
                "max_pages": chatbot[5],
                "brand_voice": chatbot[6],
                "status": chatbot[8],
                "created_at": chatbot[9],
                "updated_at": chatbot[10],
                "total_conversations": chatbot[12],
                "monthly_conversations": chatbot[13],
                "api_key": chatbot[15],
                "embed_code": chatbot[11],
                "content_pages": content_count
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/chatbots/{chatbot_id}/embed-code")
async def get_embed_code(chatbot_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT embed_code, status FROM chatbots 
                WHERE id = ? AND user_id = ?
            """, (chatbot_id, current_user["id"]))
            
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Chatbot not found")
            
            if result[1] != 'active':
                raise HTTPException(status_code=400, detail="Chatbot is not ready yet")
            
            return {
                "embed_code": result[0],
                "instructions": "Copy and paste this code into your website's HTML where you want the chatbot to appear."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.put("/chatbots/{chatbot_id}/widget-settings")
async def update_widget_settings(
    chatbot_id: int,
    settings: WidgetSettings,
    current_user: dict = Depends(get_current_user)
):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Chatbot not found")
            
            cursor.execute("""
                UPDATE chatbots 
                SET widget_settings = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(settings.dict()), chatbot_id))
            
            conn.commit()
            
            return {"message": "Widget settings updated successfully"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chatbots/{chatbot_id}")
async def delete_chatbot(
    chatbot_id: int,
    current_user: dict = Depends(get_current_user)
):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Chatbot not found or you do not have permission to delete it.")

            cursor.execute("DELETE FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
            cursor.execute("DELETE FROM conversations WHERE chatbot_id = ?", (chatbot_id,))
            cursor.execute("DELETE FROM chatbots WHERE id = ?", (chatbot_id,))
            
            conn.commit()
            
            logger.info(f"Chatbot {chatbot_id} and all associated data deleted successfully by user {current_user['id']}.")
            
            return {"message": "Chatbot and its data deleted successfully"}
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting chatbot {chatbot_id}: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred during deletion.")

# FIXED: Handle OPTIONS requests properly for CORS
@app.options("/api/chat/{chatbot_id}")
async def handle_options(chatbot_id: int):
    """Handle CORS preflight requests"""
    return JSONResponse(
        status_code=200,
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/api/chat/{chatbot_id}")
async def public_api_chat(
    chatbot_id: int, 
    chat_data: ChatMessage,
    request: Request,
    api_key: str = None
):
    start_time = datetime.now()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT c.system_prompt, c.status, c.api_key, c.user_id, c.monthly_conversations,
                       u.plan FROM chatbots c
                JOIN users u ON c.user_id = u.id
                WHERE c.id = ? AND c.is_active = 1
            """, (chatbot_id,))
            
            result = cursor.fetchone()
            if not result or result[1] != "active":
                raise HTTPException(status_code=404, detail="Chatbot not found or not active")
            
            system_prompt, status, stored_api_key, user_id, monthly_conversations, user_plan = result
            
            if api_key != stored_api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            plan_limits = SUBSCRIPTION_PLANS.get(user_plan, SUBSCRIPTION_PLANS["free"])
            if (plan_limits["max_conversations_per_month"] != -1 and 
                monthly_conversations >= plan_limits["max_conversations_per_month"]):
                raise HTTPException(status_code=429, detail="Monthly conversation limit exceeded")
            
            cursor.execute("""
                SELECT user_message, bot_response FROM conversations 
                WHERE chatbot_id = ? AND session_id = ? 
                ORDER BY created_at DESC LIMIT 10
            """, (chatbot_id, chat_data.session_id))
            
            conversation_history = [
                {"user_message": row[0], "bot_response": row[1]} 
                for row in cursor.fetchall()
            ]
            conversation_history.reverse()
            
            bot_response = await generate_ai_response(
                chat_data.message, 
                system_prompt, 
                conversation_history
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "")
            
            cursor.execute("""
                INSERT INTO conversations (chatbot_id, session_id, user_message, bot_response, 
                                            response_time_ms, user_ip, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chatbot_id, chat_data.session_id, chat_data.message, bot_response, 
                  int(response_time), client_ip, user_agent))
            
            new_monthly_count = update_conversation_count(chatbot_id, conn)
            
            cursor.execute("""
                INSERT INTO analytics (chatbot_id, event_type, event_data)
                VALUES (?, ?, ?)
            """, (chatbot_id, "api_conversation", json.dumps({
                "session_id": chat_data.session_id,
                "message_length": len(chat_data.message),
                "response_time_ms": int(response_time),
                "user_ip": client_ip,
                "monthly_count": new_monthly_count
            })))
            
            conn.commit()
            
            return {
                "response": bot_response,
                "session_id": chat_data.session_id,
                "response_time_ms": int(response_time)
            }
        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Error in API chat for chatbot {chatbot_id}: {e}")
            raise HTTPException(status_code=500, detail="Error generating response")

@app.get("/widget.js", response_class=HTMLResponse)
async def get_widget_script():
    """Serve the chatbot widget JavaScript"""
    # Assuming some dynamic values could be here, making it an f-string
    some_dynamic_value = "Custom Title" 
    widget_js = f'''
(function() {{
    window.ChatbotWidget = {{
        init: function(config) {{
            var chatbotId = config.chatbotId;
            var apiKey = config.apiKey;
            var apiUrl = config.apiUrl;
            
            // Create widget HTML - note the escaped curly braces for CSS styles
            var widgetHtml = `
                <div id="chatbot-widget" style="position: fixed; bottom: 20px; right: 20px; z-index: 10000;">
                    <div id="chatbot-button" style="width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                        <svg width="24" height="24" fill="white" viewBox="0 0 24 24">
                            <path d="M12 2C6.48 2 2 6.48 2 12c0 1.54.36 3.04.97 4.37L1 23l6.63-1.97C9.96 21.64 11.46 22 12 22c5.52 0 10-4.48 10-10S17.52 2 12 2zm0 18c-1.21 0-2.35-.31-3.34-.85L4 20l.85-4.66C4.31 14.35 4 13.21 4 12c0-4.41 3.59-8 8-8s8 3.59 8 8-3.59 8-8 8z"/>
                        </svg>
                    </div>
                    <div id="chatbot-window" style="display: none; position: absolute; bottom: 80px; right: 0; width: 350px; height: 500px; background: white; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); flex-direction: column;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px 10px 0 0; font-weight: bold;">
                            Chatbot Assistant ({some_dynamic_value})
                        </div>
                        <div id="chat-messages" style="flex: 1; padding: 15px; overflow-y: auto; max-height: 350px;">
                            <div style="background: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                                Hello! How can I help you today?
                            </div>
                        </div>
                        <div style="padding: 15px; border-top: 1px solid #eee;">
                            <div style="display: flex; gap: 10px;">
                                <input id="chat-input" type="text" placeholder="Type your message..." style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; outline: none;">
                                <button id="send-button" style="background: #667eea; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer;">➤</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Add widget to page
            document.body.insertAdjacentHTML('beforeend', widgetHtml);
            
            // Widget functionality
            var button = document.getElementById('chatbot-button');
            var window = document.getElementById('chatbot-window');
            var input = document.getElementById('chat-input');
            var sendButton = document.getElementById('send-button');
            var messages = document.getElementById('chat-messages');
            var sessionId = 'session-' + Date.now();
            
            button.onclick = function() {{
                window.style.display = window.style.display === 'none' ? 'flex' : 'none';
            }};
            
            function addMessage(message, isUser) {{
                var messageDiv = document.createElement('div');
                messageDiv.style.cssText = 'padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; ' + 
                    (isUser ? 'background: #667eea; color: white; margin-left: auto; text-align: right;' : 'background: #f0f0f0;');
                messageDiv.textContent = message;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }}
            
            function sendMessage() {{
                var message = input.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                input.value = '';
                
                // Show typing indicator
                var typingDiv = document.createElement('div');
                typingDiv.style.cssText = 'padding: 10px; border-radius: 10px; margin-bottom: 10px; background: #f0f0f0; font-style: italic;';
                typingDiv.textContent = 'Typing...';
                typingDiv.id = 'typing-indicator';
                messages.appendChild(typingDiv);
                messages.scrollTop = messages.scrollHeight;
                
                // Send to API
                fetch(apiUrl + '/api/chat/' + chatbotId + '?api_key=' + apiKey, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        message: message,
                        session_id: sessionId
                    }})
                }})
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('typing-indicator').remove();
                    if (data.response) {{
                        addMessage(data.response, false);
                    }} else {{
                        addMessage('Sorry, I encountered an error. Please try again.', false);
                    }}
                }})
                .catch(error => {{
                    document.getElementById('typing-indicator').remove();
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }});
            }}
            
            sendButton.onclick = sendMessage;
            input.onkeypress = function(e) {{
                if (e.key === 'Enter') {{
                    sendMessage();
                }}
            }};
        }}
    }};
}})();
'''
    return Response(widget_js, media_type="application/javascript")



# Analytics endpoints
@app.get("/analytics/{chatbot_id}")
async def get_chatbot_analytics(
    chatbot_id: int,
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Chatbot not found")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    AVG(response_time_ms) as avg_response_time,
                    DATE(created_at) as conversation_date
                FROM conversations 
                WHERE chatbot_id = ? AND created_at >= ? 
                GROUP BY DATE(created_at)
                ORDER BY conversation_date DESC
            """, (chatbot_id, start_date.isoformat()))
            
            daily_stats = []
            for row in cursor.fetchall():
                daily_stats.append({
                    "date": row[3],
                    "total_conversations": row[0],
                    "unique_sessions": row[1],
                    "avg_response_time_ms": round(row[2], 2) if row[2] else 0
                })
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(DISTINCT user_ip) as unique_visitors
                FROM conversations 
                WHERE chatbot_id = ? AND created_at >= ?
            """, (chatbot_id, start_date.isoformat()))
            
            overall_stats = cursor.fetchone()
            
            cursor.execute("""
                SELECT user_message, COUNT(*) as frequency
                FROM conversations 
                WHERE chatbot_id = ? AND created_at >= ?
                GROUP BY LOWER(SUBSTR(user_message, 1, 50))
                ORDER BY frequency DESC LIMIT 10
            """, (chatbot_id, start_date.isoformat()))
            
            popular_topics = []
            for row in cursor.fetchall():
                popular_topics.append({
                    "topic": row[0][:50] + "..." if len(row[0]) > 50 else row[0],
                    "frequency": row[1]
                })
            
            return {
                "period_days": days,
                "overall": {
                    "total_conversations": overall_stats[0],
                    "unique_sessions": overall_stats[1],
                    "avg_response_time_ms": round(overall_stats[2], 2) if overall_stats[2] else 0,
                    "unique_visitors": overall_stats[3]
                },
                "daily_stats": daily_stats,
                "popular_topics": popular_topics
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/subscription/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return {
        "plans": SUBSCRIPTION_PLANS,
        "current_features": {
            "free": ["1 Chatbot", "10 Pages", "100 Conversations/month", "Basic Support"],
            "starter": ["3 Chatbots", "50 Pages", "1,000 Conversations/month", "Priority Support", "Custom Styling"],
            "professional": ["10 Chatbots", "200 Pages", "5,000 Conversations/month", "Advanced Analytics", "API Access"],
            "enterprise": ["Unlimited Chatbots", "1,000 Pages", "Unlimited Conversations", "White-label", "Custom Integration"]
        }
    }

@app.post("/subscription/upgrade")
async def upgrade_subscription(
    plan_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user subscription (placeholder for payment integration)"""
    new_plan = plan_data.get("plan")
    
    if new_plan not in SUBSCRIPTION_PLANS:
        raise HTTPException(status_code=400, detail="Invalid subscription plan")
    
    new_plan_limits = SUBSCRIPTION_PLANS.get(new_plan)
    if new_plan_limits["max_chatbots"] != -1:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chatbots WHERE user_id = ?", (current_user["id"],))
            current_chatbot_count = cursor.fetchone()[0]
            if current_chatbot_count > new_plan_limits["max_chatbots"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"You currently have {current_chatbot_count} chatbots. Please delete some before downgrading to the {new_plan.capitalize()} plan which allows a maximum of {new_plan_limits['max_chatbots']}."
                )

    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            subscription_expires = datetime.now() + timedelta(days=30)
            
            cursor.execute("""
                UPDATE users 
                SET plan = ?, subscription_expires = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_plan, subscription_expires.isoformat(), current_user["id"]))
            
            conn.commit()
            
            plan_info = SUBSCRIPTION_PLANS[new_plan]
            email_body = f"""
            <html>
              <body>
                <h2>Subscription Upgraded!</h2>
                <p>Your subscription has been upgraded to {plan_info['name']}.</p>
                <p>Your new limits:</p>
                <ul>
                <li>Chatbots: {'Unlimited' if plan_info['max_chatbots'] == -1 else plan_info['max_chatbots']}</li>
                <li>Pages per bot: {plan_info['max_pages_per_bot']}</li>
                <li>Monthly conversations: {'Unlimited' if plan_info['max_conversations_per_month'] == -1 else plan_info['max_conversations_per_month']}</li>
                </ul>
                <p>Thank you for upgrading!</p>
                </body>
            </html>
            """
            
            send_email(current_user["email"], "Subscription Updated - Chatbot SaaS", email_body, is_html=True)
            
            logger.info(f"User {current_user['id']} updated plan to {new_plan}")
            
            return {
                "message": f"Successfully updated to {plan_info['name']}",
                "plan": new_plan,
                "expires": subscription_expires.isoformat()
            }
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/contact/send")
async def send_contact_form(form_data: ContactForm, background_tasks: BackgroundTasks):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contact_messages
                (name, email, subject, message, user_id, company_name, status)
            VALUES (?, ?, ?, ?, ?, ?, 'new')
        """, (
            form_data.name,
            form_data.email,
            form_data.subject,
            form_data.message,
            form_data.user_id,
            getattr(form_data, "company_name", None)
        ))
        conn.commit()

    email_html = f"""
    <html>
      <body>
        <h2>New Contact Form Submission</h2>
        <p><b>Name:</b> {form_data.name}</p>
        <p><b>Email:</b> {form_data.email}</p>
        <p><b>Subject:</b> {form_data.subject}</p>
        <p><b>Message:</b><br/>{form_data.message.replace('\n', '<br/>')}</p>
      </body>
    </html>
    """

    background_tasks.add_task(
        send_email,
        CONTACT_RECIPIENT,                         
        f"Contact Form: {form_data.subject}",
        email_html,
        True
    )

    return {"message": "Message sent successfully"}

@app.get("/debug/email-config")
def debug_email_config():
    return {
        "FROM_EMAIL": FROM_EMAIL,
        "CONTACT_RECIPIENT": CONTACT_RECIPIENT,
        "SMTP_SERVER": SMTP_SERVER,
        "SMTP_PORT": SMTP_PORT,
        "SMTP_USERNAME": SMTP_USERNAME,
        "has_SMTP_PASSWORD": bool(SMTP_PASSWORD),
    }

@app.get("/admin/stats")
async def get_admin_stats(current_user: dict = Depends(get_current_user)):
    """Get platform statistics (admin only - add proper auth in production)"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chatbots")
            total_chatbots = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("SELECT plan, COUNT(*) FROM users GROUP BY plan")
            users_by_plan = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT DATE(created_at), COUNT(*) 
                FROM conversations 
                WHERE created_at >= date('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY created_at DESC
            """)
            recent_activity = dict(cursor.fetchall())
            
            return {
                "total_users": total_users,
                "total_chatbots": total_chatbots,
                "total_conversations": total_conversations,
                "users_by_plan": users_by_plan,
                "recent_activity": recent_activity
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} - URL: {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {str(exc)} - URL: {request.url}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

@app.get("/metrics")
async def get_metrics():
    """Basic metrics for monitoring"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM chatbots WHERE status = 'active'")
            active_chatbots = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE DATE(created_at) = DATE('now')
            """)
            conversations_today = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT AVG(response_time_ms) FROM conversations 
                WHERE created_at >= datetime('now', '-1 hour')
            """)
            avg_response_time = cursor.fetchone()[0] or 0
            
            return {
                "active_chatbots": active_chatbots,
                "conversations_today": conversations_today,
                "avg_response_time_ms": round(avg_response_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Enhanced Chatbot SaaS API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )