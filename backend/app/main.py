from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.chat import ChatRequest, ChatResponse, PDFUploadResponse
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import uuid
from typing import Dict, Tuple

load_dotenv()

# API 키 설정
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Gemini 모델 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    temperature=0.7,
)

# 세션 관리를 위한 딕셔너리
chat_sessions: Dict[str, Tuple[ConversationalRetrievalChain, ConversationBufferMemory]] = {}
normal_chat_histories: Dict[str, list] = {}

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    temp_path = f"backend/data/temp/{file.filename}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # PDF 처리
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # 문서 전처리
        processed_docs = []
        for doc in documents:
            # 불필요한 공백 제거 및 텍스트 정규화
            text = doc.page_content.strip()
            text = ' '.join(text.split())  # 연속된 공백 제거
            doc.page_content = text
            processed_docs.append(doc)
        
        # 청크 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(processed_docs)
        
        # 임베딩 및 벡터 스토어 생성
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=f"backend/data/vector_store/{uuid.uuid4()}"
        )
        
        # 메모리 및 체인 생성
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="similarity",  # 유사도 기반 검색
                search_kwargs={"k": 8}  # 검색할 문서 수만 증가
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template="""다음은 이력서/자기소개서의 내용입니다. 주어진 문맥을 기반으로 질문에 상세히 답변해주세요.
만약 문맥에서 답을 찾을 수 없다면, "주어진 문서에서 해당 정보를 찾을 수 없습니다."라고 답변해주세요.

문맥:
{context}

질문: {question}

답변:""",
                    input_variables=["context", "question"]
                )
            },
            chain_type="stuff",
            verbose=True
        )
        
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = (chain, memory)
        
        return PDFUploadResponse(
            message="PDF uploaded and processed successfully",
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id
    
    if session_id in chat_sessions:
        # RAG 모드
        chain, memory = chat_sessions[session_id]
        try:
            # 검색된 문서 확인을 위한 로그
            docs = chain.retriever.get_relevant_documents(request.message)
            print(f"\nRetrieved {len(docs)} documents for query: {request.message}")
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(doc.page_content[:200] + "...")  # 처음 200자만 출력
            
            result = chain({
                "question": request.message,
                "chat_history": memory.chat_memory.messages
            })
            sources = [doc.page_content for doc in result["source_documents"]]
            
            # 메모리 업데이트
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(result["answer"])
            
            return ChatResponse(answer=result["answer"], sources=sources)
        except Exception as e:
            print(f"Error in RAG mode: {str(e)}")  # 디버깅을 위한 로그
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # 일반 채팅 모드
        if session_id not in normal_chat_histories:
            normal_chat_histories[session_id] = []
        
        chat_history = normal_chat_histories[session_id]
        
        # 대화 기록을 포함한 프롬프트 구성
        prompt = "당신은 도움이 되는 AI 어시스턴트입니다.\n\n"
        for msg, resp in chat_history:
            prompt += f"Human: {msg}\nAssistant: {resp}\n"
        prompt += f"Human: {request.message}\nAssistant:"
        
        response = await llm.ainvoke(input=prompt)
        response_text = response.content  # AIMessage에서 텍스트 추출
        
        # 대화 기록 업데이트
        chat_history.append((request.message, response_text))
        normal_chat_histories[session_id] = chat_history[-5:]  # 최근 5개 대화만 유지
        
        # 첫 메시지일 경우 session_id를 포함하여 반환
        if not request.session_id:
            return ChatResponse(
                answer=response_text,
                session_id=session_id
            )
        return ChatResponse(answer=response_text)

# 서버 종료 시 임시 파일들 정리
@app.on_event("shutdown")
async def cleanup():
    import shutil
    if os.path.exists("backend/data/temp"):
        shutil.rmtree("backend/data/temp")
    if os.path.exists("backend/data/vector_store"):
        shutil.rmtree("backend/data/vector_store") 