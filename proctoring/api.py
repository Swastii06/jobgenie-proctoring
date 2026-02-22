# Proctoring REST API Endpoints for JobGenie Integration
# This file provides JSON API endpoints that JobGenie (Next.js) can call

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.core.files.base import ContentFile
import json
import base64
import numpy as np
import cv2
import face_recognition
import logging

from .models import Student, Exam, CheatingEvent, CheatingImage
from .views import get_face_encoding, match_face_encodings

logger = logging.getLogger(__name__)


# ==================== STUDENT MANAGEMENT ====================

@csrf_exempt
def api_register_student(request):
    """
    API endpoint for student registration with face capture.
    Expects JSON POST with: name, email, password, address, photo_data (base64)
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        address = data.get('address', '')
        captured_photo = data.get('photo_data')

        # Validate required fields
        if not all([name, email, password, captured_photo]):
            return JsonResponse({"success": False, "error": "Missing required fields"}, status=400)

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return JsonResponse({"success": False, "error": "Email already registered"}, status=400)

        # Decode and process image
        try:
            img_data = base64.b64decode(captured_photo.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract face encoding
            face_encoding = get_face_encoding(image)
            if face_encoding is None:
                return JsonResponse({"success": False, "error": "No face detected in photo"}, status=400)
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Image processing error: {str(e)}"}, status=400)

        # Create user and student
        try:
            user = User.objects.create(
                username=email,
                email=email,
                first_name=name.split(' ')[0],
                last_name=' '.join(name.split(' ')[1:]) if ' ' in name else '',
                password=password,  # Will be hashed by Django
            )
            user.set_password(password)  # Properly hash the password
            user.save()

            student = Student(
                user=user,
                name=name,
                address=address,
                email=email,
                photo=ContentFile(img_data, name=f"{name}_photo.jpg"),
                face_encoding=face_encoding.tolist(),
            )
            student.save()

            return JsonResponse({
                "success": True,
                "message": "Student registered successfully",
                "student_id": student.id,
                "user_id": user.id
            }, status=201)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Database error: {str(e)}"}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


@csrf_exempt
def api_verify_student_face(request):
    """
    API endpoint for student face verification during login.
    Expects JSON POST with: email, password, photo_data (base64)
    Returns: student_id, name, email on success
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        email = data.get('email')
        password = data.get('password')
        captured_photo_data = data.get('photo_data')

        if not all([email, password, captured_photo_data]):
            return JsonResponse({"success": False, "error": "Missing required fields"}, status=400)

        # Authenticate user
        try:
            user = User.objects.get(email=email)
            if not user.check_password(password):
                return JsonResponse({"success": False, "error": "Invalid credentials"}, status=401)
        except User.DoesNotExist:
            return JsonResponse({"success": False, "error": "User not found"}, status=401)

        # Process captured photo
        try:
            captured_photo_data_clean = captured_photo_data.split(',')[1] if ',' in captured_photo_data else captured_photo_data
            img_bytes = base64.b64decode(captured_photo_data_clean)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            captured_encoding = get_face_encoding(image)
            if captured_encoding is None:
                return JsonResponse({"success": False, "error": "No face detected"}, status=400)

            # Get student and compare faces
            student = user.student
            stored_encoding = np.array(student.face_encoding)

            if match_face_encodings(captured_encoding, stored_encoding):
                return JsonResponse({
                    "success": True,
                    "message": "Face verified",
                    "student_id": student.id,
                    "name": student.name,
                    "email": student.email
                }, status=200)
            else:
                return JsonResponse({"success": False, "error": "Face does not match"}, status=401)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Face verification error: {str(e)}"}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


# ==================== EXAM MANAGEMENT ====================

@csrf_exempt
def api_start_exam(request):
    """
    API endpoint to start an exam session.
    Expects JSON POST with: student_id (Clerk ID), exam_name, total_questions
    Returns: exam_id, session_token
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        clerk_id = data.get('student_id')  # Clerk user ID
        exam_name = data.get('exam_name')
        total_questions = data.get('total_questions')
        student_name = data.get('name', 'Student')
        student_email = data.get('email', f'{clerk_id}@student.local')

        if not all([clerk_id, exam_name, total_questions]):
            return JsonResponse({"success": False, "error": "Missing required fields"}, status=400)

        try:
            # Get or create student by Clerk ID
            student, created = Student.objects.get_or_create(
                clerk_id=clerk_id,
                defaults={
                    'name': student_name,
                    'email': student_email,
                }
            )
            
            exam = Exam.objects.create(
                student=student,
                exam_name=exam_name,
                total_questions=total_questions,
                status='ongoing'
            )

            return JsonResponse({
                "success": True,
                "message": "Exam started",
                "exam_id": exam.id,
                "student_id": student.id,
                "clerk_id": student.clerk_id,
                "exam_name": exam.exam_name
            }, status=201)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Student error: {str(e)}"}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


@csrf_exempt
def api_submit_exam(request):
    """
    API endpoint to submit an exam.
    Expects JSON POST with: exam_id, correct_answers, violations_data (optional)
    Returns: result summary
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        exam_id = data.get('exam_id')
        correct_answers = data.get('correct_answers')
        violations_data = data.get('violations', [])

        if not exam_id or correct_answers is None:
            return JsonResponse({"success": False, "error": "Missing required fields"}, status=400)

        try:
            exam = Exam.objects.get(id=exam_id)
            exam.correct_answers = correct_answers
            exam.status = 'completed'
            exam.calculate_percentage()
            exam.save()

            # Save violations if any
            if violations_data:
                for violation in violations_data:
                    CheatingEvent.objects.create(
                        student=exam.student,
                        cheating_flag=True,
                        event_type=violation.get('type'),
                        detected_objects=violation.get('objects', []),
                        tab_switch_count=violation.get('tab_switches', 0)
                    )

            return JsonResponse({
                "success": True,
                "message": "Exam submitted",
                "exam_id": exam.id,
                "score": exam.percentage_score,
                "total_questions": exam.total_questions,
                "correct_answers": exam.correct_answers
            }, status=200)

        except Exam.DoesNotExist:
            return JsonResponse({"success": False, "error": "Exam not found"}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


@csrf_exempt
def api_get_exam_result(request, exam_id):
    """
    API endpoint to get exam results and violations.
    Returns: score, violations, cheating events
    """
    if request.method != "GET":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        exam = Exam.objects.get(id=exam_id)
        violations = CheatingEvent.objects.filter(student=exam.student)

        violations_data = [{
            "id": v.id,
            "type": v.event_type,
            "timestamp": v.timestamp.isoformat(),
            "objects": v.detected_objects,
            "tab_switches": v.tab_switch_count
        } for v in violations]

        return JsonResponse({
            "success": True,
            "exam": {
                "id": exam.id,
                "name": exam.exam_name,
                "status": exam.status,
                "total_questions": exam.total_questions,
                "correct_answers": exam.correct_answers,
                "score": exam.percentage_score,
                "timestamp": exam.timestamp.isoformat()
            },
            "violations": violations_data
        }, status=200)

    except Exam.DoesNotExist:
        return JsonResponse({"success": False, "error": "Exam not found"}, status=404)


@csrf_exempt
def api_record_violation(request):
    """
    API endpoint to record a proctoring violation in real-time.
    Expects JSON POST with: exam_id, student_id (Clerk ID), event_type, detected_objects, image_data (optional)
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        clerk_id = data.get('student_id')  # Clerk user ID
        event_type = data.get('event_type')
        detected_objects = data.get('detected_objects', [])
        image_data = data.get('image_data')

        if not all([clerk_id, event_type]):
            return JsonResponse({"success": False, "error": "Missing required fields"}, status=400)

        try:
            # Get or create student by Clerk ID
            student, created = Student.objects.get_or_create(
                clerk_id=clerk_id,
                defaults={'name': 'Student', 'email': f'{clerk_id}@student.local'}
            )
            
            event = CheatingEvent.objects.create(
                student=student,
                cheating_flag=True,
                event_type=event_type,
                detected_objects=detected_objects
            )

            # Save image if provided
            if image_data:
                try:
                    img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                    CheatingImage.objects.create(
                        event=event,
                        image=ContentFile(img_bytes, name=f"violation_{event.id}.jpg")
                    )
                except Exception as e:
                    logger.warning(f"Failed to save violation image: {str(e)}")

            return JsonResponse({
                "success": True,
                "message": "Violation recorded",
                "event_id": event.id
            }, status=201)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Error: {str(e)}"}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


@csrf_exempt
def api_record_tab_switch(request):
    """
    API endpoint to record tab switches during exam.
    Expects JSON POST with: student_id (Clerk ID), count
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        clerk_id = data.get('student_id')  # Clerk user ID
        tab_switch_count = data.get('count', 1)

        if not clerk_id:
            return JsonResponse({"success": False, "error": "Missing student_id"}, status=400)

        try:
            # Get or create student by Clerk ID
            student, created = Student.objects.get_or_create(
                clerk_id=clerk_id,
                defaults={'name': 'Student', 'email': f'{clerk_id}@student.local'}
            )
            
            # Find the latest ongoing exam
            exam = Exam.objects.filter(student=student, status='ongoing').latest('timestamp')
            
            event = CheatingEvent.objects.create(
                student=student,
                cheating_flag=True,
                event_type='tab_switch',
                tab_switch_count=tab_switch_count
            )

            return JsonResponse({
                "success": True,
                "message": "Tab switch recorded",
                "event_id": event.id
            }, status=201)

        except Student.DoesNotExist:
            return JsonResponse({"success": False, "error": "Student not found"}, status=404)
        except Exam.DoesNotExist:
            return JsonResponse({"success": False, "error": "No ongoing exam found"}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)


@csrf_exempt
def api_health_check(request):
    """
    Simple health check endpoint to verify the proctoring service is running.
    """
    return JsonResponse({
        "success": True,
        "message": "Proctoring API is running",
        "service": "JobGenie Proctoring System"
    }, status=200)


@csrf_exempt
def api_analyze_frame(request):
    """
    Analyze a video frame for proctoring violations.
    Expects JSON POST with: frame_data (base64), exam_id, user_id
    Returns: violations array with detected issues
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        frame_data = data.get('frame_data')
        exam_id = data.get('exam_id')
        user_id = data.get('user_id')

        if not frame_data:
            return JsonResponse({"success": False, "error": "Frame data required"}, status=400)

        # Decode base64 frame
        try:
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({
                    "success": True,
                    "violations": [],
                    "no_face_detected": False,
                    "mobile_devices_detected": False
                })
        except Exception as e:
            logger.error(f"Frame decode error: {str(e)}")
            return JsonResponse({
                "success": True,
                "violations": [],
                "no_face_detected": False
            })

        # Prepare defaults
        violations = []
        no_face_detected = False
        multiple_faces_detected = False
        face_blocked = False
        head_pose_angle = 0

        # ========== FACE DETECTION ==========
        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_count = len(face_locations)

            # Check for no face
            if face_count == 0:
                violations.append({
                    "type": "NO_FACE",
                    "message": "No face detected in frame",
                    "severity": "medium"
                })
                no_face_detected = True
            else:
                no_face_detected = False

            # Check for multiple faces
            if face_count > 1:
                violations.append({
                    "type": "MULTIPLE_FACES",
                    "message": f"Multiple faces detected ({face_count})",
                    "severity": "high",
                    "confidence": 90
                })
                multiple_faces_detected = True
            else:
                multiple_faces_detected = False

            # When exactly one face is present, run more detailed checks using landmarks
            if face_count == 1:
                top, right, bottom, left = face_locations[0]
                face_width = right - left
                face_height = bottom - top
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]

                # Check if face is partially occluded (too close to edge)
                margin_threshold = 30
                if left < margin_threshold or right > (frame_width - margin_threshold) or \
                   top < margin_threshold or bottom > (frame_height - margin_threshold):
                    violations.append({
                        "type": "FACE_BLOCKED",
                        "message": "Face too close to frame edge",
                        "severity": "medium"
                    })
                    face_blocked = True
                else:
                    face_blocked = False

                # Estimate head pose angle from face center offset
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                frame_center_x = frame_width / 2
                frame_center_y = frame_height / 2
                
                # Horizontal angle (left-right turn)
                angle_x = abs(face_center_x - frame_center_x) / frame_width * 90
                # Vertical angle (up-down tilt)
                angle_y = abs(face_center_y - frame_center_y) / frame_height * 60
                head_pose_angle = int(max(angle_x, angle_y))
                
                # Flag if face is turned too far (> 25 degrees)
                if head_pose_angle > 25:
                    violations.append({
                        "type": "HEAD_POSE",
                        "message": f"Face turned too far (angle: {head_pose_angle}°)",
                        "severity": "medium",
                        "head_pose_angle": head_pose_angle
                    })

                # Use facial landmarks to check eyes and occlusion
                landmarks_list = face_recognition.face_landmarks(rgb_frame)
                if landmarks_list:
                    landmarks = landmarks_list[0]

                    def compute_ear(eye_points):
                        xs = [p[0] for p in eye_points]
                        ys = [p[1] for p in eye_points]
                        width = max(xs) - min(xs) if max(xs) - min(xs) > 0 else 1
                        height = max(ys) - min(ys)
                        return float(height) / float(width)

                    # Simple occlusion check: missing nose_bridge or nose_tip points indicates possible occlusion
                    if not landmarks.get('nose_bridge') or not landmarks.get('nose_tip'):
                        # If nose landmarks missing, likely occluded (hand or object)
                        face_blocked = True
                        violations.append({
                            "type": "FACE_BLOCKED",
                            "message": "Face partially occluded",
                            "severity": "high"
                        })

                else:
                    # Landmarks not found despite face location - mark possible occlusion
                    face_blocked = True
                    violations.append({
                        "type": "FACE_BLOCKED",
                        "message": "Could not extract facial landmarks; possible occlusion",
                        "severity": "medium"
                    })

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            # ensure defaults remain set
            no_face_detected = no_face_detected if 'no_face_detected' in locals() else False
            multiple_faces_detected = multiple_faces_detected if 'multiple_faces_detected' in locals() else False
            face_blocked = face_blocked if 'face_blocked' in locals() else False
            head_pose_angle = head_pose_angle if 'head_pose_angle' in locals() else 0

        # ========== OBJECT DETECTION (YOLO) ==========
        suspicious_objects = []
        try:
            try:
                from ultralytics import YOLO
                # Try to load default YOLO model (will download if not present)
                model = YOLO('yolov8n.pt')
                results = model(frame, verbose=False, conf=0.25)
                
                # Check for suspicious objects
                suspicious_labels = ['cell phone', 'phone', 'mobile phone', 'laptop', 'book', 'document', 'paper', 'cup', 'watch', 'hand']
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        label = result.names.get(class_id, f'class_{class_id}').lower()
                        confidence = float(box.conf)
                        
                        if any(suspicious in label for suspicious in suspicious_labels):
                            suspicious_objects.append(label)
                            violations.append({
                                "type": "SUSPICIOUS_OBJECT",
                                "message": f"Detected: {label}",
                                "severity": "high",
                                "confidence": int(confidence * 100)
                            })
            except ImportError:
                # YOLO not installed, try simple color-based phone detection
                # Phones often have rectangular contours and smooth surfaces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Phone-like rectangular object in mid-frame
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.4 < aspect_ratio < 2.5 and w > 30 and h > 50:
                        suspicious_objects.append('potential_phone')
                        violations.append({
                            "type": "SUSPICIOUS_OBJECT",
                            "message": "Potential phone/device detected",
                            "severity": "high",
                            "confidence": 60
                        })
                        break
        except Exception as e:
            logger.debug(f"Object detection error: {str(e)}")
            suspicious_objects = []

        # ========== EYES DETECTION (landmarks-only) ==========
        # Eyes closed detection is already done above using face_recognition landmarks
        # Disable Haar cascade eye detection as it causes too many false positives

        return JsonResponse({
            "success": True,
            "violations": violations,
            "no_face_detected": no_face_detected,
            "multiple_faces_detected": multiple_faces_detected,
            "face_blocked": face_blocked,
            "head_pose_angle": head_pose_angle,
            "suspicious_objects": suspicious_objects,
            "violation_count": len(violations),
            "timestamp": str(np.datetime64('now'))
        })
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Analyze frame error: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": str(e),
            "violations": []
        }, status=500)

