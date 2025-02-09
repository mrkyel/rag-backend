# RAG Chatbot Project

PDF 문서 기반 RAG(Retrieval Augmented Generation) 챗봇 프로젝트입니다. PDF 파일을 업로드하여 문서 내용에 대해 질문하거나, 일반적인 대화를 할 수 있습니다.

## 주요 기능

- 📄 PDF 파일 업로드 및 문서 기반 질의응답
- 💬 일반 채팅 모드 지원
- 🔄 멀티턴 대화 지원
- 📝 대화 기록 유지
- 🔍 문서 내용 검색 및 참조

## 기술 스택

### Backend

- FastAPI
- LangChain
- Google Gemini Pro
- Chroma DB (벡터 저장소)
- HuggingFace Embeddings

### Frontend

- Next.js
- TypeScript
- TailwindCSS

## 시작하기

### 사전 요구사항

- Python 3.11 이상
- Node.js 18 이상
- Google AI Studio API 키

### 설치 방법

1. 저장소 클론

git clone https://github.com/yourusername/rag-chatbot.git

cd rag-chatbot

2. 백엔드 설정
   cd backend
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. 환경 변수 설정
   backend/.env 파일 생성
   GOOGLE_API_KEY=your_google_api_key_here

4. 프론트엔드 설정

npx를 활용하여 개별 생성 (ex: next, cra etc)

### 실행 방법

1. 백엔드 서버 실행

cd backend
uvicorn app.main:app --reload

## 사용 방법

1. 일반 채팅

   - 웹사이트 접속 후 바로 채팅 시작
   - AI 어시스턴트와 자유롭게 대화

2. PDF 기반 채팅
   - PDF 파일 업로드
   - 업로드된 문서 내용에 대해 질문
   - 관련 내용을 찾아 답변 제공

## API 엔드포인트

- `GET /health`: 서버 상태 확인
- `POST /api/upload`: PDF 파일 업로드
- `POST /api/chat`: 채팅 메시지 전송
