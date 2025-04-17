import os
import pandas as pd
import re
import sqlite3
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PromptAnalyzer:
    def __init__(self, database_path: str = "prompts.db"):
        """Initialize the PromptAnalyzer with database connection."""
        self.database_path = database_path
        self.df = None
        self.setup_database()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Groq client
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

    def setup_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS harmful_prompts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        is_malicious BOOLEAN,
                        is_jailbreak BOOLEAN,
                        groq_confirmed BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database setup error: {e}")
            raise

    def load_data(self, dataset_path: str) -> None:
        """Load and preprocess the dataset."""
        try:
            self.df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(self.df):,} rows")
            self._preprocess_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _preprocess_data(self) -> None:
        """Clean and preprocess the data."""
        original_count = len(self.df)
        
        # Clean prompts
        self.df['Prompt_clean'] = self.df['Prompt'].str.lower().str.strip()
        
        # Remove duplicates
        duplicate_mask = self.df.duplicated(subset=['Prompt_clean'], keep='last')
        self.df = self.df[~duplicate_mask].copy()
        
        logger.info(f"Removed {original_count - len(self.df):,} duplicate rows")

    @staticmethod
    def get_malicious_keywords() -> List[str]:
        """Return list of malicious keywords."""
        return [
    # Jailbreak & Injection
    "ignore previous instructions", "you are no longer", "DAN mode", 
    "as an uncensored AI", "roleplay as", "hypothetically", 
    "bypass safety", "override rules", "for educational purposes", 
    "simulate illegal", "pretend to", "break character",

    # Deepfake & Impersonation
    "create deepfake", "fake video of", "impersonate", "clone voice", 
    "replace face", "put X's face on", "realistic fake", "deepfake tutorial",

            # NSFW / Adult
    "nude", "naked", "porn", "sex scene", "leaked video", "NSFW",
    "explicit content", "18+", "OnlyFans", "erotic", "kissing", 
    "hot girl", "seduce", "masturbate", "undress", "XXX",

            # Violence / Harm
    "shooting video", "beheading", "murder", "torture", "explosion", 
    "gore", "blood everywhere", "violent attack", "terrorist act",

            # Hate Speech
    "kill all [group]", "gas the", "racial slur", "hate speech", 
    "homophobic", "Nazi propaganda", "white power", "KKK",

    # Misinformation
    "fake news broadcast", "make it look real", "hoax", "undetectable fake", 
    "fabricate story", "lie convincingly", "official looking",

            # Child Exploitation
    "underage", "teen nude", "schoolgirl sex", "jailbait", "loli", 
    "child porn", "cp", "pedo", "minor NSFW",

    # Adversarial Attacks
    "leak prompt", "training data extraction", "bypass filter", 
    "exploit system", "prompt injection", "override safety", "Waluigi Effect"
]

    @staticmethod
    def get_jailbreak_patterns() -> List[str]:
        """Return list of jailbreak regex patterns."""
        return [
    r"(?i)(ignore|disregard) (the |all )?(previous|prior) (instructions|rules)",
    r"(?i)(you are|act as) (an? )?(uncensored|unfiltered|unethical|evil|unrestricted)",
    r"(?i)(DAN|do anything now|jailbreak|roleplay as|simulate being)",
    r"(?i)(this is a test|for educational purposes|hypothetically speaking)",
    r"(?i)(pretend to|break character|simulate illegal|override safety)",
            r"(?i)(i know you can't but|i know you're not allowed to but|try anyway)",
]

    def analyze_prompts(self) -> None:
        """Analyze prompts for harmful content."""
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")

        # Check for malicious keywords
        self.df["is_malicious"] = self.df["Prompt"].str.contains(
            "|".join(self.get_malicious_keywords()),
            case=False, regex=True
        )

        # Check for jailbreak patterns
        self.df["is_jailbreak"] = False
        for pattern in self.get_jailbreak_patterns():
            self.df["is_jailbreak"] |= self.df["Prompt"].str.contains(pattern, regex=True)

        # Combine flags
        self.df["is_harmful"] = self.df["is_malicious"] | self.df["is_jailbreak"]
        
        # Get harmful prompts
        self.harmful_prompts = self.df[self.df["is_harmful"]].copy()
        logger.info(f"Found {len(self.harmful_prompts):,} potentially harmful prompts")

    def is_malicious_groq(self, prompt: str, model: str = "deepseek-r1-distill-llama-70b") -> bool:
        """Check if a prompt is malicious using Groq API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a safety classifier. Try to label each row from the data (equally one "prompt") based on its general content, 
                        and then ranking all of them based on their most harmful/malicious prompts based on your own justification. 
                        Respond ONLY with 'True' or 'False'."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            return result == "True" if result in ["True", "False"] else False
        except Exception as e:
            logger.error(f"Groq API error for prompt '{prompt[:50]}...': {e}")
            return False

    def analyze_top_harmful_prompts(self, limit: int = 10) -> None:
        """Analyze top harmful prompts using Groq API."""
        if not hasattr(self, 'harmful_prompts'):
            raise ValueError("No harmful prompts analyzed. Run analyze_prompts first.")

        top_harmful_prompts = self.harmful_prompts.head(limit).copy()
        top_harmful_prompts["groq_confirmed"] = top_harmful_prompts["Prompt"].apply(
            lambda x: self.is_malicious_groq(x)
        )
        
        # Save to database
        self.save_to_database(top_harmful_prompts)
        
        logger.info(f"Analyzed and saved {limit} top harmful prompts")

    def save_to_database(self, df: pd.DataFrame) -> None:
        """Save analyzed prompts to database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                for _, row in df.iterrows():
                    conn.execute('''
                        INSERT INTO harmful_prompts (prompt, is_malicious, is_jailbreak, groq_confirmed)
                        VALUES (?, ?, ?, ?)
                    ''', (row['Prompt'], row['is_malicious'], row['is_jailbreak'], row['groq_confirmed']))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    def get_harmful_prompts_from_db(self) -> pd.DataFrame:
        """Retrieve harmful prompts from database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                return pd.read_sql_query("SELECT * FROM harmful_prompts", conn)
        except sqlite3.Error as e:
            logger.error(f"Database read error: {e}")
            raise

def main():
    # Initialize analyzer
    analyzer = PromptAnalyzer()
    
    # Set your dataset path
    dataset_path = os.getenv("DATASET_PATH", "/kaggle/input/prompts")
    
    try:
        # Load and analyze data
        analyzer.load_data(dataset_path)
        analyzer.analyze_prompts()
        analyzer.analyze_top_harmful_prompts(limit=10)
        
        # Retrieve results
        results = analyzer.get_harmful_prompts_from_db()
        logger.info(f"Retrieved {len(results)} harmful prompts from database")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()