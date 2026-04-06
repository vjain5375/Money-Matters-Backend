"""
MoneyMattersAI -- Financial Intelligence API
=============================================

A FastAPI-powered REST API that exposes the MoneyMattersAI ML system.

Endpoints:
    GET  /            -- System info and health check
    POST /classify    -- Classify a single transaction description
    POST /analyze     -- Analyze a list of transactions with amounts
    GET  /health      -- Liveness probe (for deployment / uptime checks)

Run locally:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Swagger UI (auto-generated):
    http://localhost:8000/docs

ReDoc UI:
    http://localhost:8000/redoc
"""

import os
import sys
import time
import logging
from datetime import datetime

# -- Resolve project root so imports work when running from any directory --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.predict import predict_transaction, load_artifacts
from src.expense_analytics import analyze_from_records

# Stock analysis router (Indian stocks — NSE/BSE)
try:
    from stock.router import router as stock_router
    _stock_router_available = True
except Exception as e:
    import traceback
    traceback.print_exc()
    _stock_router_available = False


# =========================================================================
# Logger
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MoneyMattersAI.API")


# =========================================================================
# App Initialization
# =========================================================================
app = FastAPI(
    title="MoneyMattersAI API",
    description=(
        "Financial Intelligence API that classifies expense transactions "
        "and generates spending analytics using machine learning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# -- CORS Middleware -------------------------------------------------------
# Restricted origins for production security
ALLOWED_ORIGINS = [
    "http://localhost:5173",          # Local Dev (Vite default)
    "https://moneymattersai.tech",    # Main Domain
    "https://www.moneymattersai.tech",
    "https://dashboard.moneymattersai.tech",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if os.getenv("ENV") == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Register Stock Router ------------------------------------------------
# Mounted at /stock/* — independent module, won't break if deps missing
if _stock_router_available:
    app.include_router(stock_router)
    logger.info("Stock analysis router registered at /stock/*")
else:
    logger.warning("Stock router not available — install yfinance & pandas-ta")


# =========================================================================
# Startup Event -- Pre-load ML artifacts into memory
# =========================================================================
@app.on_event("startup")
async def preload_model():
    """
    Load the ML model and TF-IDF vectorizer into memory at startup.

    Pre-loading avoids cold-start latency on the first API request.
    The artifacts are cached in predict.py's module-level variables.
    """
    logger.info("Pre-loading ML model and TF-IDF vectorizer...")
    try:
        load_artifacts(verbose=False)
        logger.info("ML artifacts loaded successfully.")
    except FileNotFoundError as e:
        logger.error(
            f"Model not found: {e}\n"
            f"Please run 'python src/train_model.py' before starting the API."
        )
    # Start keep-alive background task
    import asyncio
    asyncio.create_task(keep_alive())


async def keep_alive():
    """Ping /health every 10 minutes to prevent Render free tier from sleeping."""
    import httpx
    import asyncio
    await asyncio.sleep(60)  # wait 1 min after startup
    while True:
            # Use RENDER_EXTERNAL_URL or fallback to the current Render subdomain
            render_url = os.getenv("RENDER_EXTERNAL_URL") or "https://money-matters-backend-pc4i.onrender.com"
            async with httpx.AsyncClient() as client:
                await client.get(f"{render_url}/health", timeout=10)
            logger.info(f"Keep-alive ping sent to {render_url}")
        except Exception as e:
            logger.error(f"Keep-alive ping failed: {str(e)}")
        await asyncio.sleep(600)  # ping every 10 minutes


# =========================================================================
# Request Timing Middleware
# =========================================================================
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to every response."""
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


# =========================================================================
# Pydantic Models (Request / Response Schemas)
# =========================================================================

# -- /classify schemas ----------------------------------------------------
class ClassifyRequest(BaseModel):
    transaction: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw transaction description to classify.",
        example="zomato dinner",
    )

    @validator("transaction")
    def transaction_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Transaction text cannot be blank.")
        return v.strip()


class ClassifyResponse(BaseModel):
    transaction: str = Field(..., description="Original transaction text.")
    category: str = Field(..., description="Predicted expense category.")
    confidence: float = Field(..., description="Model confidence score (0–1).")
    all_probabilities: dict = Field(
        ..., description="Probability distribution across all categories."
    )
    cleaned_text: str = Field(..., description="Preprocessed transaction text.")


# -- /analyze schemas -----------------------------------------------------
class TransactionRecord(BaseModel):
    transaction: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Transaction description.",
        example="zomato dinner",
    )
    amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount in INR (must be >= 0).",
        example=350.0,
    )
    category: str = Field(
        default=None,
        description=(
            "Optional expense category. If omitted, the ML model classifies it."
        ),
        example="Food",
    )

    @validator("transaction")
    def transaction_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Transaction text cannot be blank.")
        return v.strip()


class AnalyticsResponse(BaseModel):
    total_spending: float
    total_transactions: int
    category_breakdown: dict
    percentages: dict
    insights: list
    top_transactions: list
    category_details: dict


# -- /get-advice schemas --------------------------------------------------
class AdviceRequest(BaseModel):
    expenses: dict = Field(
        ...,
        description="Category-to-amount spending dict.",
        example={"Food": 3500, "Shopping": 8000, "Transport": 1200},
    )


class AdviceResponse(BaseModel):
    advice: str = Field(..., description="AI-generated financial advice.")


# =========================================================================
# ENDPOINT 1: GET / -- System Info
# =========================================================================
@app.get(
    "/",
    summary="System Info",
    description="Returns system metadata and available module list.",
    tags=["System"],
)
async def root():
    """
    GET /

    Returns system information about the MoneyMattersAI platform.

    Example response:
        {
            "system": "MoneyMatters AI",
            "version": "1.0",
            "modules": ["Expense Classifier", "Expense Analytics"]
        }
    """
    return {
        "system": "MoneyMatters AI",
        "version": "1.0",
        "description": "AI-powered financial intelligence platform.",
        "modules": [
            "Expense Classifier",
            "Expense Analytics",
            "Stock Analysis (NSE/BSE)",
            "Finance LLaMA Advisor",
        ],
        "endpoints": {
            "classify":   "POST /classify    -- Classify a single transaction",
            "analyze":    "POST /analyze     -- Analyze a batch of transactions",
            "get_advice": "POST /get-advice  -- AI financial advice from LLaMA",
            "health":     "GET  /health      -- API health check",
            "docs":       "GET  /docs        -- Swagger UI",
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# =========================================================================
# ENDPOINT 2: GET /health -- Liveness Probe
# =========================================================================
@app.get(
    "/health",
    summary="Health Check",
    tags=["System"],
)
async def health_check():
    """
    GET /health

    Liveness probe — confirms the service is running and the ML model
    is loaded. Useful for deployment monitoring and container health checks.
    """
    try:
        model, vectorizer = load_artifacts(verbose=False)
        return {
            "status": "healthy",
            "model": type(model).__name__,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "categories": list(model.classes_),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ML model not loaded. "
                "Run 'python src/train_model.py' to train the model first."
            ),
        )


# =========================================================================
# ENDPOINT 3: POST /classify -- Classify a Single Transaction
# =========================================================================
@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify Transaction",
    description="Predicts the expense category for a single transaction description.",
    tags=["ML Endpoints"],
)
async def classify_transaction(request: ClassifyRequest):
    """
    POST /classify

    Classifies a single transaction description into an expense category.

    Request body:
        { "transaction": "zomato dinner" }

    Response:
        {
            "transaction": "zomato dinner",
            "category": "Food",
            "confidence": 0.39,
            "all_probabilities": { "Food": 0.39, "Shopping": 0.21, ... },
            "cleaned_text": "zomato dinner"
        }
    """
    logger.info(f"POST /classify  [{request.transaction!r}]")

    try:
        result = predict_transaction(request.transaction, verbose=False)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not found: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    logger.info(
        f"  -> Category: {result['predicted_category']} "
        f"(confidence: {result['confidence']:.2%})"
    )

    return ClassifyResponse(
        transaction=result["transaction"],
        category=result["predicted_category"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        cleaned_text=result["cleaned_text"],
    )


# =========================================================================
# ENDPOINT 4: POST /analyze -- Analyze a Batch of Transactions
# =========================================================================
@app.post(
    "/analyze",
    response_model=AnalyticsResponse,
    summary="Analyze Transactions",
    description=(
        "Accepts a JSON list of transactions with amounts, classifies them "
        "using the ML model, and returns a full spending analytics report."
    ),
    tags=["ML Endpoints"],
)
async def analyze_transactions(transactions: list[TransactionRecord]):
    """
    POST /analyze

    Analyzes a batch of transactions and returns a spending report.

    Request body (list of transaction objects):
        [
            { "transaction": "zomato dinner", "amount": 350 },
            { "transaction": "uber ride",     "amount": 220 },
            { "transaction": "amazon order",  "amount": 1200 }
        ]

    Response:
        {
            "total_spending": 1770,
            "category_breakdown": { "Food": 350, "Transport": 220, ... },
            "percentages": { "Food": 20, "Transport": 12, ... },
            "insights": [ "Food is your highest ...", ... ],
            ...
        }
    """
    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Transaction list cannot be empty.",
        )

    logger.info(f"POST /analyze  [{len(transactions)} transactions]")

    # Convert Pydantic models to plain dicts for the analytics engine
    records = [
        {
            "transaction": t.transaction,
            "amount": t.amount,
            **({"category": t.category} if t.category else {}),
        }
        for t in transactions
    ]

    try:
        report = analyze_from_records(records)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid data: {str(e)}",
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not found: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )

    logger.info(
        f"  -> Total: Rs.{report['total_spending']:,.2f} | "
        f"Categories: {list(report['category_breakdown'].keys())}"
    )

    return AnalyticsResponse(**report)


# =========================================================================
# ENDPOINT 5: POST /get-advice -- Finance LLaMA AI Advisor
# =========================================================================
@app.post(
    "/get-advice",
    response_model=AdviceResponse,
    summary="AI Financial Advice",
    description=(
        "Generates personalised financial advice using the fine-tuned "
        "Finance LLaMA (LLaMA-3-8B + LoRA) model based on a spending breakdown."
    ),
    tags=["LLM Endpoints"],
)
async def get_advice(request: AdviceRequest):
    """
    POST /get-advice

    Pass a category-to-amount dict and receive AI-generated financial advice.

    Request body:
        { "expenses": {"Food": 3500, "Shopping": 8000, "Transport": 1200} }

    Response:
        { "advice": "You are spending 53% on Shopping which is quite high..." }

    Note: First call loads the LLaMA model (~160MB adapters) — expect 30-60s cold start.
    """
    logger.info(f"POST /get-advice  [expenses: {list(request.expenses.keys())}]")

    if not request.expenses:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Expenses dict cannot be empty.",
        )

    try:
        # Import lazily so the server starts fast even without GPU
        import importlib.util
        import sys as _sys
        advisor_path = os.path.join(
            os.path.dirname(__file__), "..", "llm", "inference", "advisor.py"
        )
        # Use a cached module reference to avoid reloading the 8B model every call
        if "_finance_advisor" not in _sys.modules:
            spec = importlib.util.spec_from_file_location("_finance_advisor", advisor_path)
            advisor_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(advisor_mod)
            _sys.modules["_finance_advisor"] = advisor_mod
        else:
            advisor_mod = _sys.modules["_finance_advisor"]

        advice = advisor_mod.generate_advice(request.expenses)
        logger.info(f"  -> Advice generated ({len(advice)} chars)")
        return AdviceResponse(advice=advice)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLaMA adapter not found: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advice generation failed: {str(e)}",
        )


# =========================================================================
# Global Exception Handler
# =========================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred.",
            "detail": str(exc),
        },
    )


# =========================================================================
# Dev Entry Point
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
