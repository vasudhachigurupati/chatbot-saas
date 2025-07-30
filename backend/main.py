import os
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import warnings
import logging
import json
import re
import xml.etree.ElementTree as ET

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Form, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from groq import Groq
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
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        company_name TEXT NOT NULL,
        plan TEXT NOT NULL DEFAULT 'free',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Chatbots table
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
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_trained TIMESTAMP,
        total_conversations INTEGER NOT NULL DEFAULT 0,
        settings TEXT NOT NULL DEFAULT '{}',
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
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id)
    )
    """)
    
    # Conversations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chatbot_id INTEGER NOT NULL,
        session_id TEXT,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        response_time_ms INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id)
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
        FOREIGN KEY (chatbot_id) REFERENCES chatbots (id)
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    company_name: str
    plan: str = Field(default="free")
    
    @field_validator('plan')
    @classmethod
    def validate_plan(cls, v):
        if v not in ['free', 'premium', 'enterprise']:
            raise ValueError('Plan must be one of: free, premium, enterprise')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatbotCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    website_url: str
    sitemap_url: Optional[str] = None
    max_pages: int = Field(default=10, ge=1, le=100)
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

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatbotUpdate(BaseModel):
    name: Optional[str] = None
    brand_voice: Optional[str] = None
    max_pages: Optional[int] = None

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
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        "id": user[0],
        "email": user[1],
        "company_name": user[3],
        "plan": user[4],
        "is_active": user[5]
    }

# Optimized WebScraper Class
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
            
            # Fallback to body if no main content found
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

# AI Response Generation
def generate_system_prompt(brand_voice: str, website_content: List[Dict]) -> str:
    """Generate system prompt based on content and brand voice"""
    voice_instructions = {
        "friendly": "You are a friendly and approachable assistant. Use a warm, conversational tone and be helpful and enthusiastic.",
        "professional": "You are a professional assistant. Use formal language, be precise and informative, and maintain a business-appropriate tone.",
        "casual": "You are a casual and relaxed assistant. Use informal language, be conversational and easy-going.",
        "technical": "You are a technical expert. Use precise terminology, provide detailed explanations, and focus on accuracy and technical depth."
    }
    
    # Summarize website content (top 5 pages)
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
        
        # Add recent conversation history
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

# Background task for website processing
async def process_website(chatbot_id: int, chatbot_data: ChatbotCreate):
    """Background task to scrape website and set up chatbot"""
    try:
        logger.info(f"Starting to process chatbot {chatbot_id}")
        
        async with WebScraper(max_pages=chatbot_data.max_pages) as scraper:
            # Get URLs to scrape (unified method)
            urls_to_scrape = await scraper.get_urls_to_scrape(
                chatbot_data.website_url, 
                chatbot_data.sitemap_url
            )
            
            if not urls_to_scrape:
                raise Exception("No URLs could be discovered for scraping")
            
            # Scrape content from URLs
            scraped_content = await scraper.scrape_content_from_urls(urls_to_scrape)
            
            if not scraped_content:
                raise Exception("No content could be scraped from the website")
        
        # Save to database
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        try:
            # Save scraped content
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
            
            # Generate and save system prompt
            system_prompt = generate_system_prompt(chatbot_data.brand_voice, scraped_content)
            
            cursor.execute("""
                UPDATE chatbots 
                SET system_prompt = ?, status = 'active', updated_at = CURRENT_TIMESTAMP, last_trained = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (system_prompt, chatbot_id))
            
            conn.commit()
            logger.info(f"✅ Chatbot {chatbot_id} processed successfully with {len(scraped_content)} pages")
            
        except Exception as db_error:
            logger.error(f"Database error for chatbot {chatbot_id}: {db_error}")
            raise
        finally:
            conn.close()
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error processing chatbot {chatbot_id}: {error_msg}")
        
        # Update status to failed
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE chatbots 
                SET status = 'failed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chatbot_id,))
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.error(f"Failed to update chatbot status: {db_error}")

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Universal Website Chatbot API is starting up...")
    init_database()
    logger.info("📊 Database initialized successfully!")
    yield
    # Shutdown
    logger.info("👋 Shutting down...")

# FastAPI app
app = FastAPI(
    title="Universal Website Chatbot API",
    description="A comprehensive SaaS platform for creating intelligent chatbots from any website",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request/response logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# API Routes

@app.get("/")
async def root():
    return {
        "message": "🤖 Universal Website Chatbot API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        # Test database connection
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "version": "2.0.0"
    }

# Authentication endpoints
@app.post("/auth/register", response_model=dict)
async def register(user_data: UserCreate):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        cursor.execute("""
            INSERT INTO users (email, hashed_password, company_name, plan)
            VALUES (?, ?, ?, ?)
        """, (user_data.email, hashed_password, user_data.company_name, user_data.plan))
        
        user_id = cursor.lastrowid
        conn.commit()
        
        logger.info(f"New user registered: {user_data.email} (ID: {user_id})")
        
        return {
            "message": "User created successfully",
            "user_id": user_id,
            "email": user_data.email
        }
    finally:
        conn.close()

@app.post("/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, email, hashed_password FROM users WHERE email = ?", (username,))
        user = cursor.fetchone()
        
        if not user or not verify_password(password, user[2]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        access_token = create_access_token(data={"sub": str(user[0]), "email": user[1]})
        
        logger.info(f"User logged in: {user[1]} (ID: {user[0]})")
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    finally:
        conn.close()

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return current_user

# Chatbot management endpoints
@app.post("/chatbots/create")
async def create_chatbot(
    chatbot_data: ChatbotCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
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
        
        # Start background processing
        background_tasks.add_task(process_website, chatbot_id, chatbot_data)
        
        logger.info(f"Chatbot created: {chatbot_data.name} (ID: {chatbot_id}) for user {current_user['id']}")
        
        return {
            "id": chatbot_id,
            "name": chatbot_data.name,
            "status": "processing",
            "message": "Chatbot creation started. Website scraping in progress..."
        }
    finally:
        conn.close()

@app.get("/chatbots/list")
async def list_chatbots(current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, name, website_url, status, created_at, 
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
                "page_count": row[5]
            })
        
        return chatbots
    finally:
        conn.close()

@app.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM chatbots 
            WHERE id = ? AND user_id = ?
        """, (chatbot_id, current_user["id"]))
        
        chatbot = cursor.fetchone()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Get content count
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
            "content_pages": content_count
        }
    finally:
        conn.close()

@app.put("/chatbots/{chatbot_id}")
async def update_chatbot(
    chatbot_id: int,
    chatbot_update: ChatbotUpdate,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify ownership
        cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Build dynamic update query
        update_fields = []
        values = []
        
        if chatbot_update.name is not None:
            update_fields.append("name = ?")
            values.append(chatbot_update.name)
        
        if chatbot_update.brand_voice is not None:
            update_fields.append("brand_voice = ?")
            values.append(chatbot_update.brand_voice)
        
        if chatbot_update.max_pages is not None:
            update_fields.append("max_pages = ?")
            values.append(chatbot_update.max_pages)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(chatbot_id)
        
        query = f"UPDATE chatbots SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)
        conn.commit()
        
        logger.info(f"Chatbot {chatbot_id} updated by user {current_user['id']}")
        
        return {"message": "Chatbot updated successfully"}
    finally:
        conn.close()

@app.delete("/chatbots/{chatbot_id}")
async def delete_chatbot(chatbot_id: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify ownership
        cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Delete related data (cascading delete)
        cursor.execute("DELETE FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
        cursor.execute("DELETE FROM conversations WHERE chatbot_id = ?", (chatbot_id,))
        cursor.execute("DELETE FROM analytics WHERE chatbot_id = ?", (chatbot_id,))
        cursor.execute("DELETE FROM chatbots WHERE id = ?", (chatbot_id,))
        
        conn.commit()
        
        logger.info(f"Chatbot {chatbot_id} deleted by user {current_user['id']}")
        
        return {"message": "Chatbot deleted successfully"}
    finally:
        conn.close()

@app.post("/chatbots/{chatbot_id}/regenerate")
async def regenerate_chatbot(
    chatbot_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Get chatbot details
        cursor.execute("""
            SELECT name, website_url, sitemap_url, max_pages, brand_voice
            FROM chatbots 
            WHERE id = ? AND user_id = ?
        """, (chatbot_id, current_user["id"]))
        
        chatbot = cursor.fetchone()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Clear existing content
        cursor.execute("DELETE FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
        
        # Update status to processing
        cursor.execute("""
            UPDATE chatbots 
            SET status = 'processing', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chatbot_id,))
        
        conn.commit()
        
        # Create chatbot data for reprocessing
        chatbot_data = ChatbotCreate(
            name=chatbot[0],
            website_url=chatbot[1],
            sitemap_url=chatbot[2],
            max_pages=chatbot[3],
            brand_voice=chatbot[4]
        )
        
        # Start background reprocessing
        background_tasks.add_task(process_website, chatbot_id, chatbot_data)
        
        logger.info(f"Chatbot {chatbot_id} regeneration started by user {current_user['id']}")
        
        return {
            "message": "Chatbot regeneration started. Website re-scraping in progress...",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error regenerating chatbot {chatbot_id}: {e}")
        
        # Update status to failed
        cursor.execute("""
            UPDATE chatbots 
            SET status = 'failed', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chatbot_id,))
        conn.commit()
        
        raise HTTPException(status_code=500, detail="Error regenerating chatbot")
    finally:
        conn.close()

# Chat endpoints
@app.post("/chat/{chatbot_id}")
async def chat_with_bot(
    chatbot_id: int,
    chat_data: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    start_time = datetime.now()
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify chatbot ownership and get system prompt
        cursor.execute("""
            SELECT system_prompt, status FROM chatbots 
            WHERE id = ? AND user_id = ?
        """, (chatbot_id, current_user["id"]))
        
        chatbot = cursor.fetchone()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        if chatbot[1] != "active":
            raise HTTPException(status_code=400, detail=f"Chatbot is not ready (status: {chatbot[1]})")
        
        system_prompt = chatbot[0]
        
        # Get recent conversation history
        cursor.execute("""
            SELECT user_message, bot_response FROM conversations 
            WHERE chatbot_id = ? AND session_id = ? 
            ORDER BY created_at DESC LIMIT 10
        """, (chatbot_id, chat_data.session_id))
        
        conversation_history = [
            {"user_message": row[0], "bot_response": row[1]} 
            for row in cursor.fetchall()
        ]
        conversation_history.reverse()  # Chronological order
        
        # Generate AI response
        bot_response = await generate_ai_response(
            chat_data.message, 
            system_prompt, 
            conversation_history
        )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save conversation
        cursor.execute("""
            INSERT INTO conversations (chatbot_id, session_id, user_message, bot_response, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (chatbot_id, chat_data.session_id, chat_data.message, bot_response, int(response_time)))
        
        # Log analytics event
        cursor.execute("""
            INSERT INTO analytics (chatbot_id, event_type, event_data)
            VALUES (?, ?, ?)
        """, (chatbot_id, "message_sent", json.dumps({
            "session_id": chat_data.session_id,
            "message_length": len(chat_data.message),
            "response_time_ms": int(response_time),
            "user_id": current_user["id"]
        })))
        
        conn.commit()
        
        return {
            "response": bot_response,
            "session_id": chat_data.session_id,
            "response_time_ms": int(response_time)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat for chatbot {chatbot_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")
    finally:
        conn.close()

@app.post("/public/chat/{chatbot_id}")
async def public_chat(chatbot_id: int, chat_data: ChatMessage):
    """Public endpoint for chatbot usage without authentication"""
    start_time = datetime.now()
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Get chatbot system prompt and verify it's active
        cursor.execute("""
            SELECT system_prompt, status FROM chatbots 
            WHERE id = ? AND is_active = 1
        """, (chatbot_id,))
        
        chatbot = cursor.fetchone()
        if not chatbot or chatbot[1] != "active":
            raise HTTPException(status_code=404, detail="Chatbot not found or not active")
        
        system_prompt = chatbot[0]
        
        # Get recent conversation history for this session
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
        
        # Generate AI response
        bot_response = await generate_ai_response(
            chat_data.message, 
            system_prompt, 
            conversation_history
        )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save conversation
        cursor.execute("""
            INSERT INTO conversations (chatbot_id, session_id, user_message, bot_response, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (chatbot_id, chat_data.session_id, chat_data.message, bot_response, int(response_time)))
        
        # Log analytics event
        cursor.execute("""
            INSERT INTO analytics (chatbot_id, event_type, event_data)
            VALUES (?, ?, ?)
        """, (chatbot_id, "public_message_sent", json.dumps({
            "session_id": chat_data.session_id,
            "message_length": len(chat_data.message),
            "response_time_ms": int(response_time)
        })))
        
        conn.commit()
        
        return {
            "response": bot_response,
            "session_id": chat_data.session_id,
            "response_time_ms": int(response_time)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in public chat for chatbot {chatbot_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")
    finally:
        conn.close()

@app.get("/chat/{chatbot_id}/history")
async def get_chat_history(
    chatbot_id: int,
    session_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify chatbot ownership
        cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Get conversation history
        cursor.execute("""
            SELECT user_message, bot_response, created_at, response_time_ms
            FROM conversations 
            WHERE chatbot_id = ? AND session_id = ? 
            ORDER BY created_at DESC LIMIT ?
        """, (chatbot_id, session_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "user_message": row[0],
                "bot_response": row[1],
                "timestamp": row[2],
                "response_time_ms": row[3]
            })
        
        conversations.reverse()  # Chronological order
        
        return {
            "conversations": conversations,
            "total": len(conversations)
        }
    finally:
        conn.close()

# Analytics and content endpoints
@app.get("/analytics/{chatbot_id}")
async def get_chatbot_analytics(
    chatbot_id: int,
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify chatbot ownership
        cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily conversation metrics
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
        
        # Get overall metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_conversations,
                COUNT(DISTINCT session_id) as unique_sessions,
                AVG(response_time_ms) as avg_response_time
            FROM conversations 
            WHERE chatbot_id = ? AND created_at >= ?
        """, (chatbot_id, start_date.isoformat()))
        
        overall_stats = cursor.fetchone()
        
        # Get popular topics (based on message content)
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
                "avg_response_time_ms": round(overall_stats[2], 2) if overall_stats[2] else 0
            },
            "daily_stats": daily_stats,
            "popular_topics": popular_topics
        }
    finally:
        conn.close()

@app.get("/chatbots/{chatbot_id}/content")
async def get_chatbot_content(
    chatbot_id: int,
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Verify chatbot ownership
        cursor.execute("SELECT id FROM chatbots WHERE id = ? AND user_id = ?", (chatbot_id, current_user["id"]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Get website content
        cursor.execute("""
            SELECT url, title, content, metadata, crawled_at
            FROM website_content 
            WHERE chatbot_id = ? 
            ORDER BY crawled_at DESC
        """, (chatbot_id,))
        
        content_pages = []
        for row in cursor.fetchall():
            metadata = json.loads(row[3]) if row[3] else {}
            content_pages.append({
                "url": row[0],
                "title": row[1],
                "content_preview": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "word_count": metadata.get("word_count", 0),
                "crawled_at": row[4]
            })
        
        return {
            "pages": content_pages,
            "total_pages": len(content_pages)
        }
    finally:
        conn.close()

# Debug endpoints (remove in production)
@app.get("/debug/chatbot/{chatbot_id}")
async def debug_chatbot(chatbot_id: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Get chatbot details
        cursor.execute("""
            SELECT id, name, website_url, sitemap_url, status, system_prompt, created_at, updated_at
            FROM chatbots 
            WHERE id = ? AND user_id = ?
        """, (chatbot_id, current_user["id"]))
        
        chatbot = cursor.fetchone()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        # Get scraped content count
        cursor.execute("SELECT COUNT(*) FROM website_content WHERE chatbot_id = ?", (chatbot_id,))
        content_count = cursor.fetchone()[0]
        
        # Get recent conversations
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE chatbot_id = ?", (chatbot_id,))
        conversation_count = cursor.fetchone()[0]
        
        return {
            "chatbot_id": chatbot[0],
            "name": chatbot[1],
            "website_url": chatbot[2],
            "sitemap_url": chatbot[3],
            "status": chatbot[4],
            "has_system_prompt": bool(chatbot[5]),
            "system_prompt_preview": chatbot[5][:200] + "..." if chatbot[5] else None,
            "created_at": chatbot[6],
            "updated_at": chatbot[7],
            "scraped_pages": content_count,
            "total_conversations": conversation_count
        }
    finally:
        conn.close()

# Error handlers
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

# CORS options handler
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    return JSONResponse(status_code=200, content={})

# Run the application
if __name__ == "__main__":
    logger.info("Starting Universal Website Chatbot API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )