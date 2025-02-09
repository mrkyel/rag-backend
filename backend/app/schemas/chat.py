from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # 선택적 필드로 변경

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    session_id: Optional[str] = None  # 새로 추가

class Document(BaseModel):
    text: str
    metadata: dict = {}

class DocumentUploadRequest(BaseModel):
    documents: List[Document]

class DocumentUploadResponse(BaseModel):
    message: str
    count: int

class PDFUploadResponse(BaseModel):
    message: str
    session_id: str  # PDF 업로드 시 새로운 세션 ID 생성 