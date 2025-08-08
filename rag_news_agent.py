import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Fix for NumPy 2.0 compatibility issue
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

class RAGNewsAgent:
    def __init__(self):
        """Initialize RAG-powered news agent with vector database capabilities"""
        
        # Initialize LLM with free DeepSeek model
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            temperature=0.1,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=2000
        )
        
        # Initialize embeddings for vector creation (RAG component)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        # Initialize text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Setup vector database for RAG
        self.setup_vector_store()
        self.setup_tools()
        self.setup_agent()

    def setup_vector_store(self):
        """Setup ChromaDB vector database for persistent storage"""
        try:
            self.persist_directory = "./rag_vector_db"
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB with persistent storage
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="news_articles_rag"
            )
            print("✅ Vector store initialized successfully")
        except Exception as e:
            print(f"⚠️ Vector store setup failed: {e}")
            self.vector_store = None

    def generate_search_keywords(self, industry: str) -> list:
        """Generate enhanced search keywords for any industry"""
        industry_clean = industry.strip().lower()
        
        # Enhanced keyword mapping for better search results
        industry_keywords = {
            "sports": ["football", "basketball", "soccer", "baseball", "tennis", "golf", "olympics", "NFL", "NBA", "cricket"],
            "technology": ["AI", "tech", "software", "startup", "innovation", "cyber", "digital", "machine learning"],
            "finance": ["banking", "fintech", "cryptocurrency", "bitcoin", "stock market", "trading", "investment"],
            "healthcare": ["medical", "pharma", "biotech", "health", "medicine", "clinical", "FDA", "vaccine"],
            "fashion": ["fashion", "clothing", "apparel", "luxury brands", "style", "designer", "retail fashion"],
            "automotive": ["cars", "electric vehicles", "Tesla", "auto", "EV", "autonomous vehicles"],
            "energy": ["renewable energy", "solar", "wind", "oil", "gas", "electricity", "power"],
            "entertainment": ["movies", "gaming", "streaming", "media", "Netflix", "entertainment industry"],
            "ev": ["electric vehicles", "tesla", "battery", "charging", "automotive", "clean energy", "electric cars"]
        }
        
        # Get industry-specific keywords
        specific_keywords = industry_keywords.get(industry_clean, [])
        
        # Build comprehensive keyword list
        keywords = [industry_clean, f"{industry_clean} industry", f"{industry_clean} news"]
        keywords.extend(specific_keywords)
        
        # Add generic variations
        words = industry_clean.split()
        if len(words) == 1:
            base_word = words[0]
            keywords.extend([
                f"{base_word} sector", f"{base_word} market",
                f"{base_word} business", f"{base_word} companies"
            ])
        
        # Add search variations
        keywords.extend([
            f"{industry_clean} trends", f"{industry_clean} developments",
            f"{industry_clean} analysis", f"{industry_clean} report"
        ])
        
        return list(set(keywords))

    def fetch_thenews_api(self, industry: str, num_articles: int = 10) -> dict:
        """Enhanced news fetching with multiple search strategies"""
        api_key = os.getenv("THENEWS_API_KEY")
        if not api_key:
            return {"error": "TheNews API key not found"}
        
        keywords = self.generate_search_keywords(industry)
        
        # Multiple search strategies for better coverage
        search_strategies = [
            " OR ".join(keywords[:5]),  # Top 5 keywords
            f"{industry} OR {industry} industry OR {industry} news",
            " OR ".join([kw for kw in keywords if len(kw) > 3])[:150],  # Avoid too long queries
            keywords[0],  # Primary keyword
            f'"{industry}"'  # Exact phrase
        ]
        
        all_articles = []
        
        for strategy_idx, query in enumerate(search_strategies):
            if len(all_articles) >= num_articles:
                break
                
            url = "https://api.thenewsapi.com/v1/news/all"
            params = {
                "api_token": api_key,
                "search": query,
                "language": "en",
                "limit": 15,  # Request more to filter duplicates
                "published_after": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "sort": "published_desc"
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("data", [])
                    
                    # Filter for relevance and add unique articles
                    relevant_articles = self.filter_relevant_articles(articles, industry, keywords)
                    existing_urls = {article.get("url") for article in all_articles}
                    
                    for article in relevant_articles:
                        if article.get("url") not in existing_urls and len(all_articles) < num_articles:
                            all_articles.append(article)
                            existing_urls.add(article.get("url"))
                            
            except Exception as e:
                continue
        
        if all_articles:
            return {
                "articles": all_articles[:num_articles],
                "total_found": len(all_articles)
            }
        else:
            return {"error": f"No relevant news found for '{industry}'"}

    def filter_relevant_articles(self, articles: list, industry: str, keywords: list) -> list:
        """Filter articles for relevance to the industry"""
        relevant_articles = []
        industry_terms = [industry.lower()] + [kw.lower() for kw in keywords[:10]]
        
        for article in articles:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            
            # Check if any industry terms appear in title or description
            relevance_score = 0
            for term in industry_terms:
                if term in title:
                    relevance_score += 2  # Title match is more important
                elif term in description:
                    relevance_score += 1
            
            # Include articles with relevance score > 0
            if relevance_score > 0:
                article["relevance_score"] = relevance_score
                relevant_articles.append(article)
        
        # Sort by relevance score (highest first)
        relevant_articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_articles

    def store_in_vector_db(self, articles: list, industry: str) -> int:
        """Store articles in vector database for RAG functionality"""
        if not self.vector_store or not articles:
            return 0
        
        try:
            documents = []
            for article in articles:
                # Create rich content for better retrieval
                content = f"""
Title: {article.get('title', 'No title')}
Description: {article.get('description', 'No description')}
Industry: {industry}
Published: {article.get('published_at', 'Unknown')}
Source: {article.get('source', 'Unknown')}
URL: {article.get('url', '')}
"""
                
                # Split into chunks for better vector storage
                chunks = self.text_splitter.split_text(content.strip())
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "title": article.get('title', 'No title'),
                            "url": article.get('url', ''),
                            "industry": industry,
                            "published_at": article.get('published_at', ''),
                            "source": str(article.get('source', 'Unknown')),
                            "chunk_id": f"{article.get('url', 'no_url')}_{i}",
                            "storage_date": datetime.now().isoformat()
                        }
                    ))
            
            if documents:
                # Add to vector database
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                return len(documents)
                
        except Exception as e:
            print(f"Vector storage error: {e}")
        return 0

    def get_rag_context(self, industry: str, k: int = 5) -> list:
        """Retrieve relevant context from vector database (RAG retrieval)"""
        if not self.vector_store:
            return []
        
        try:
            # Semantic search for relevant historical articles
            results = self.vector_store.similarity_search(
                f"{industry} industry news trends",
                k=k,
                filter={"industry": industry}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in results
            ]
        except:
            return []

    def format_articles_clean(self, articles: list, industry: str) -> str:
        """Format articles in clean, professional style"""
        if not articles:
            return f"No articles found for {industry}."
        
        result = ""
        for i, article in enumerate(articles[:10], 1):
            title = article.get("title", "No title available")
            description = article.get("description", "No description available")
            url = article.get("url", "#")
            pub_date = article.get("published_at", "Unknown date")
            
            # Format date
            try:
                if pub_date and pub_date != "Unknown date":
                    parsed_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    formatted_date = parsed_date.strftime("%B %d, %Y at %I:%M %p")
                else:
                    formatted_date = pub_date
            except:
                formatted_date = pub_date
            
            # Get source
            source = article.get("source", "Unknown source")
            if isinstance(source, dict):
                source_name = source.get("name", "Unknown source")
            else:
                source_name = str(source)
            
            # Format article
            result += f"**📌 Article {i}: {title}**\n\n"
            
            if description and description != "No description available":
                desc_clean = description[:200] + "..." if len(description) > 200 else description
                result += f"📝 **Summary:** {desc_clean}\n\n"
            
            result += f"📅 **Published:** {formatted_date}\n"
            result += f"📰 **Source:** {source_name}\n"
            result += f"🔗 **Read More:** {url}\n\n"
        
        return result

    def analyze_with_rag(self, articles: list, industry: str, rag_context: list) -> str:
        """Enhanced analysis using RAG (Retrieval-Augmented Generation)"""
        if not articles:
            return f"No analysis available for {industry}."
        
        # Prepare current articles for analysis
        current_text = ""
        for article in articles[:5]:
            title = article.get("title", "")
            desc = article.get("description", "")
            current_text += f"{title}. {desc} "
        
        # Prepare historical context from vector database
        historical_context = ""
        if rag_context:
            historical_context = "Historical context: " + " ".join([
                ctx["content"][:150] for ctx in rag_context[:3]
            ])
        
        # RAG-enhanced prompt
        prompt = f"""Analyze the {industry} industry using both current news and historical context:

CURRENT NEWS:
{current_text}

HISTORICAL CONTEXT (from vector database):
{historical_context}

Provide analysis in this format:
📊 **Sentiment:** [Positive/Negative/Mixed]
📈 **Market Outlook:** [Brief outlook based on current + historical data]
🎯 **Key Trends:** [2-3 main trends from current news]
📚 **Historical Insights:** [How current trends relate to past patterns if available]
💡 **Strategic Insights:** [Actionable recommendations]

Keep it concise and professional."""
        
        try:
            ai_response = self.llm.invoke(prompt)
            return ai_response.content
        except:
            return f"✅ RAG analysis: Found {len(articles)} current articles and {len(rag_context)} historical references for {industry}."

    def get_rag_industry_analysis(self, industry: str) -> str:
        """Main RAG analysis function - implements complete RAG pipeline"""
        
        # Step 1: Fetch fresh articles
        news_result = self.fetch_thenews_api(industry, 10)
        
        if "error" in news_result:
            return f"❌ {news_result['error']}"
        
        articles = news_result.get("articles", [])
        
        # Step 2: Store articles in vector database (RAG storage)
        stored_chunks = self.store_in_vector_db(articles, industry)
        
        # Step 3: Retrieve historical context (RAG retrieval)
        rag_context = self.get_rag_context(industry, 5)
        
        # Step 4: Format articles
        formatted_articles = self.format_articles_clean(articles, industry)
        
        # Step 5: RAG-enhanced analysis
        rag_analysis = self.analyze_with_rag(articles, industry, rag_context)
        
        # Step 6: Add RAG enhancement info
        rag_info = f"\n**🧠 RAG Enhancement:**\n"
        rag_info += f"- Stored {stored_chunks} new chunks in vector database\n"
        rag_info += f"- Retrieved {len(rag_context)} historical articles for context\n"
        rag_info += f"- Enhanced analysis with historical patterns\n"
        
        # Combine all results
        result = formatted_articles + "\n" + rag_analysis + rag_info
        return result

    def setup_tools(self):
        """Setup tools for LangChain agent"""
        self.tools = [
            Tool(
                name="GetRAGIndustryNews",
                func=self.get_rag_industry_analysis,
                description="Get 10 articles with RAG-enhanced analysis for any industry."
            )
        ]

    def setup_agent(self):
        """Setup LangChain agent with RAG capabilities"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional RAG-powered news analyst. You use:
            1. Fresh news articles from multiple sources
            2. Historical context from vector database
            3. Enhanced analysis combining current + historical data
            
            Always provide 10 recent articles with RAG-enhanced insights."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,  # Increased from 2 to 5
            max_execution_time=60,  # Add timeout (60 seconds)
            early_stopping_method="generate"  # Return partial results if needed
        )

    def analyze(self, industry: str) -> str:
        """Main analysis method with RAG enhancement"""
        try:
            response = self.agent_executor.invoke({
                "input": f"Get 10 recent articles with RAG-enhanced analysis for {industry} industry",
                "chat_history": []
            })
            return response["output"]
        except Exception as e:
            # Fallback to direct analysis if agent fails
            print(f"Agent execution failed, using direct analysis: {e}")
            return self.get_rag_industry_analysis(industry)

def main():
    """Clean main function"""
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        return
    
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("THENEWS_API_KEY"):
        print("❌ API keys not found in .env file!")
        return
    
    try:
        agent = RAGNewsAgent()
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return
    
    # Clean CLI loop
    while True:
        industry = input("Enter industry: ").strip()
        
        if industry.lower() == 'quit':
            print("Goodbye!")
            break
        elif not industry:
            continue
        
        print(f"\nAnalyzing {industry}...\n")
        result = agent.analyze(industry)
        print(result)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
