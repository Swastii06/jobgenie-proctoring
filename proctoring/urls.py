from django.urls import path
from . import views
from . import api  # Import API endpoints
from django.conf import settings
from django.conf.urls.static import static
from .views import home, registration,login,video_feed,dashboard,exam_submission_success,exam,submit_exam,result,record_tab_switch,get_warning,add_question
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('registration/', views.registration, name='registration'),
    path('login/', views.login, name='login'),
    path('video_feed/', views.video_feed, name='video_feed'),  # For video feed
    # path('stop_event /', views.stop_event , name='stop_event'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('exam/', views.exam, name='exam'),
    path('submit_exam/', views.submit_exam, name='submit_exam'),
    path('exam_submission_success/', views.exam_submission_success, name='exam_submission_success'),
    path('result/', views.result, name='result'),
    path('get_warning/', views.get_warning, name='get_warning'),
    path('proctor_notifications/', views.proctor_notifications, name='proctor_notifications'),
    path('record_tab_switch/', views.record_tab_switch, name='record_tab_switch'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('report_page/<int:student_id>/', views.report_page, name='report_page'),
    path('logout/',views.logout, name='logout'),
    path('download_report/<int:student_id>/', views.download_report, name='download_report'),
    path('admin_dashboard/add_question/', add_question, name='add_question'),

    # ============ JOBGENIE PROCTORING API ENDPOINTS ============
    path('api/health/', api.api_health_check, name='api_health_check'),
    path('api/register-student/', api.api_register_student, name='api_register_student'),
    path('api/verify-face/', api.api_verify_student_face, name='api_verify_face'),
    path('api/start-exam/', api.api_start_exam, name='api_start_exam'),
    path('api/submit-exam/', api.api_submit_exam, name='api_submit_exam'),
    path('api/exam-result/<int:exam_id>/', api.api_get_exam_result, name='api_exam_result'),
    path('api/record-violation/', api.api_record_violation, name='api_record_violation'),
    path('api/record-tab-switch/', api.api_record_tab_switch, name='api_record_tab_switch'),
    path('api/proctoring/analyze-frame/', api.api_analyze_frame, name='api_analyze_frame'),
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
