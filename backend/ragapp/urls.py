from django.urls import path

from . import views


urlpatterns = [
    path("health", views.health),
    path("upload", views.upload_document),
    path("upload-stream", views.upload_document_stream),
    path("progress/<str:job_id>", views.progress),
    path("chat", views.chat),
    path("chat-sessions", views.chat_sessions),
    path("chat-sessions/<str:session_id>", views.chat_session_detail),
    path("documents", views.documents),
    path("chunks", views.chunks),
    path("chunks-file", views.chunks_file),
    path("retrieval-logs", views.retrieval_logs),
]
