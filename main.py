from database import SessionLocal, ChatHistory, User
from auth import hash_password, verify_password, create_access_token, decode_access_token
from fastapi import Header, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from tavily import TavilyClient
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

@app.post("/register")
def register(request: AuthRequest):
    db = SessionLocal()
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        db.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(email=request.email, hashed_password=hash_password(request.password))
    db.add(new_user)
    db.commit()
    db.close()
    return {"message": "User registered successfully"}

@app.post("/login")
def login(request: AuthRequest):
    db = SessionLocal()
    user = db.query(User).filter(User.email == request.email).first()
    db.close()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"user_id": user.id, "email": user.email})
    return {"access_token": token}

@app.post("/auth/google")
def auth_google(request: GoogleAuthRequest):
    try:
        idinfo = google_id_token.verify_oauth2_token(
            request.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError as e:
        print("GOOGLE TOKEN VERIFY FAILED:", str(e))
        raise HTTPException(status_code=401, detail="Invalid Google token")

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Google account has no email")

    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        user = User(email=email, hashed_password=hash_password(os.urandom(16).hex()))
        db.add(user)
        db.commit()
        db.refresh(user)

    db.close()

    token = create_access_token({"user_id": user.id, "email": user.email})
    return {"access_token": token}

class Question(BaseModel):
    query: str

@app.get("/history")
def get_history(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("user_id")

    db = SessionLocal()
    entries = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )
    db.close()

    return [
        {
            "question": e.question,
            "answer": e.answer,
            "sources": e.sources.split(", ") if e.sources else [],
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]

@app.post("/ask/stream")
def ask_stream(question: Question, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("user_id")
    user_query = question.query

    # Marker that separates the streamed answer text from the trailing
    # JSON metadata (sources, used_web_search). Chosen to be extremely
    # unlikely to appear naturally in model output.
    SOURCES_MARKER = "\n§§SOURCES_JSON§§"

    def generate():
        sources = []
        used_web_search = False
        full_answer = ""

        try:
            decision_prompt = f"""Question: "{user_query}"
Does this need current/up-to-date internet information (news, prices, live events, recent stats)?
Reply with ONLY: YES or NO."""

            decision = model.generate_content(decision_prompt).text.strip().upper()
            used_web_search = "YES" in decision

            if used_web_search:
                search_results = tavily_client.search(query=user_query, max_results=5)

                context_text = ""
                for result in search_results["results"]:
                    context_text += f"Source: {result['url']}\nContent: {result['content']}\n\n"
                    sources.append(result["url"])

                final_prompt = f"""Based on these search results, answer the question clearly.

Question: {user_query}

Search Results:
{context_text}

Give a clear, well-organized answer using specific facts from the sources."""
            else:
                final_prompt = user_query

            final_text_for_db = ""

            try:
                stream = model.generate_content(final_prompt, stream=True)
                for chunk in stream:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                final_text_for_db = full_answer

            except Exception as stream_err:
                stream_error_message = str(stream_err)
                print("MID-STREAM ERROR:", stream_error_message)

                if "503" in stream_error_message or "overloaded" in stream_error_message.lower():
                    friendly = "The model is currently overloaded with requests. Please try again in a moment."
                elif "429" in stream_error_message or "quota" in stream_error_message.lower():
                    friendly = "You're sending requests too quickly. Please wait 30-60 seconds before asking again."
                else:
                    friendly = "The response was interrupted due to an unexpected error."

                # Clearly separated from whatever partial text already streamed,
                # instead of being silently glued onto the end of it.
                interruption_notice = f"\n\n---\n⚠️ **{friendly}**"
                full_answer += interruption_notice
                yield interruption_notice
                final_text_for_db = None  # don't save a broken/incomplete answer

            # Trailing metadata so the frontend can split it off
            yield SOURCES_MARKER + json.dumps({
                "sources": sources,
                "used_web_search": used_web_search
            })

            if final_text_for_db is not None:
                db = SessionLocal()
                new_entry = ChatHistory(
                    user_id=user_id,
                    question=user_query,
                    answer=final_text_for_db,
                    used_web_search=str(used_web_search),
                    sources=", ".join(sources)
                )
                db.add(new_entry)
                db.commit()
                db.close()

        except Exception as e:
            error_message = str(e)
            print("STREAM ERROR:", error_message)

            if "429" in error_message or "quota" in error_message.lower():
                yield "⚠️ You're sending requests too quickly. Please wait about 30-60 seconds before asking again."
            else:
                yield f"An error occurred: {error_message}"

            yield SOURCES_MARKER + json.dumps({"sources": [], "used_web_search": False})

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/")
def health_check():
    return {"status": "ResearchPilot backend is running"}

@app.post("/ask")
def ask(question: Question, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("user_id")

    print("Using key ending in:", os.getenv("GEMINI_API_KEY")[-6:])
    user_query = question.query
    sources = []

    try:
        decision_prompt = f"""Question: "{user_query}"
Does this need current/up-to-date internet information (news, prices, live events, recent stats)?
Reply with ONLY: YES or NO."""

        decision = model.generate_content(decision_prompt).text.strip().upper()

        if "YES" in decision:
            search_results = tavily_client.search(query=user_query, max_results=5)

            context_text = ""
            for result in search_results["results"]:
                context_text += f"Source: {result['url']}\nContent: {result['content']}\n\n"
                sources.append(result["url"])

            final_prompt = f"""Based on these search results, answer the question clearly.

Question: {user_query}

Search Results:
{context_text}

Give a clear, well-organized answer using specific facts from the sources."""

            final_answer = model.generate_content(final_prompt).text
        else:
            final_answer = model.generate_content(user_query).text

        db = SessionLocal()
        new_entry = ChatHistory(
            user_id=user_id,
            question=user_query,
            answer=final_answer,
            used_web_search=str("YES" in decision),
            sources=", ".join(sources)
        )
        db.add(new_entry)
        db.commit()
        db.close()

        return {
            "answer": final_answer,
            "used_web_search": "YES" in decision,
            "sources": sources
        }

    except Exception as e:
        error_message = str(e)
        print("FULL ERROR:", error_message)

        if "429" in error_message or "quota" in error_message.lower():
            return {
                "answer": "⚠️ You're sending requests too quickly. Please wait about 30-60 seconds before asking again.",
                "used_web_search": False,
                "sources": []
            }

        return {
            "answer": f"An error occurred: {error_message}",
            "used_web_search": False,
            "sources": []
        }